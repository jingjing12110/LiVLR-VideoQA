# @File: h2gr.py
# @Time: 2020/8/14
# @Email: jingjingjiang2017@gmail.com
import torch
import torch.nn as nn
from typing import Dict
from torch.nn.utils.weight_norm import weight_norm

from module.dynamic_rnn import PadLSTM
from model.classifier import OpenEndedClassifier
from model.object_relation_encoder import ObjectRelationEncoder

from model.semantic_role_encoder import SemanticRoleEncoder
from model.visual_semantic_encoder import MultiGranularityVSEncoder



def init_rnn_weights(rnn, rnn_type, num_layers=None):
    if rnn_type == 'lstm':
        if num_layers is None:
            num_layers = rnn.num_layers
        for layer in range(num_layers):
            nn.init.orthogonal_(getattr(rnn, f'weight_hh_l{layer}'))
            nn.init.orthogonal_(getattr(rnn, f'weight_ih_l{layer}'))
            # nn.init.kaiming_normal_(getattr(rnn, f'weight_ih_l{layer}'))
            nn.init.constant_(getattr(rnn, f'bias_hh_l{layer}'), val=0)
            nn.init.constant_(getattr(rnn, f'bias_ih_l{layer}'), val=0)
            getattr(rnn, f'bias_ih_l{layer}').data.index_fill_(0, torch.arange(
                rnn.hidden_size, rnn.hidden_size * 2).long(), 1)

    if rnn_type == 'bilstm':
        if num_layers is None:
            num_layers = rnn.num_layers
        for layer in range(num_layers):
            for name in ['i', 'h']:
                try:
                    weight = getattr(rnn, f'weight_{name}h_l{layer}')
                    weight_r = getattr(rnn, f'weight_{name}h_l{layer}_reverse')
                except:
                    weight = getattr(rnn, f'weight_{name}h')
                    weight_r = getattr(rnn, f'weight_{name}h_reverse')
                nn.init.orthogonal_(weight.data)
                nn.init.orthogonal_(weight_r.data)
                try:
                    bias = getattr(rnn, f'bias_{name}h_l{layer}')
                    bias_r = getattr(rnn, f'bias_{name}h_l{layer}_reverse')
                except:
                    bias = getattr(rnn, f'bias_{name}h')  # BUTD: LSTM Cell
                    bias_r = getattr(rnn, f'bias_{name}h_reverse')
                nn.init.constant_(bias, 0)
                nn.init.constant_(bias_r, 0)
                if name == 'i':
                    bias.data.index_fill_(0, torch.arange(
                        rnn.hidden_size, rnn.hidden_size * 2).long(), 1)
                    bias_r.data.index_fill_(0, torch.arange(
                        rnn.hidden_size, rnn.hidden_size * 2).long(), 1)


class H2GR(nn.Module):
    def __init__(self, cfg):
        super(H2GR, self).__init__()
        self.cfg = cfg
        self.with_root_node = True
        self.bs = None

        self.q_encoding = nn.Sequential(
            nn.LayerNorm(cfg.token_dim),
            weight_norm(nn.Linear(cfg.token_dim, cfg.hid_dim), dim=None),
            nn.LayerNorm(cfg.hid_dim),
            nn.ReLU(),
            nn.Dropout(p=cfg.dropout, inplace=True),
        )

        # image encoder
        self.img_encoder = nn.Sequential(
            nn.LayerNorm(2048),
            # nn.BatchNorm1d(64),
            weight_norm(nn.Linear(2048, cfg.hid_dim), dim=None),
            nn.LayerNorm(cfg.hid_dim),
            # nn.BatchNorm1d(64),
            # nn.ReLU(True),
            nn.ReLU(),
            nn.Dropout(p=cfg.dropout, inplace=True),
        )

        # semantic role encoder
        self.sr_encoder = SemanticRoleEncoder(
            input_dim=cfg.token_dim,
            hidden_dim=cfg.hid_dim,
            encoder_name='gcn',
            gcn_attention=False,
            dropout=cfg.dropout,
            with_root_node=self.with_root_node
        )

        # object relation encoder
        self.or_encoder = ObjectRelationEncoder(
            hidden_dim=cfg.hid_dim,
            encoder_name='gcn',
            gcn_attention=False,
            dropout=cfg.dropout,
            num_head=cfg.num_head,
            with_co_attn=False
        )

        # relation reasoning
        self.rel_reasoning = RelationReasoning(
            hidden_dim=cfg.hid_dim,
            num_head=cfg.num_head,
            dropout=cfg.dropout
        )

        # answer prediction
        self.classifier = OpenEndedClassifier(
            cfg.hid_dim * 2,
            cfg.hid_dim,
            num_answers=1000,
            dropout=cfg.dropout
        )

    def forward(self, batch_data) -> Dict:
        # ########################## feature encoder ###########################
        q_embed = self.q_encoding(  # [bs, n_token, 512]
            batch_data['q_bert'].float().cuda(non_blocking=True))
        # semantic role encoder
        sent_embed, role_embed, role_mask = self.forward_semantic_role(
            batch_data)

        # image encoder
        img_embed = self.img_encoder(  # [bs, #img, 512]
            batch_data.get('img_feat').float().cuda())

        # object relation encoder
        obj_embed = self.or_encoder(  # [bs, #img, #obj, 512]
            batch_data.get('obj_feat').float().cuda(),
            # batch_data.get('attr_feat').float().cuda(non_blocking=True),
            batch_data.get('sp_feat').float().cuda(),
            batch_data.get('obj_adj_matrix').float().cuda()
        )

        # ######################### relation reasoning ########################
        fusion_embed = self.rel_reasoning(  # [bs, 512]
            q_embed,
            batch_data['q_mask'].bool().cuda(),
            img_embed,
            obj_embed,
            sent_embed, role_embed, role_mask
        )

        logit = self.classifier(fusion_embed)

        return {
            'logit': logit
        }

    def forward_semantic_role(self, b_data):
        sent_bert = b_data['sent_bert'].float()
        self.bs = sent_bert.shape[0]
        sent_bert = sent_bert.view(
            self.bs * self.cfg.num_srl, -1, 768).cuda(non_blocking=True)
        sent_mask = b_data['sent_mask'].view(
            self.bs * self.cfg.num_srl, -1).cuda(non_blocking=True)

        node_role = b_data['node_role'].view(
            self.bs * self.cfg.num_srl, -1).long().cuda(non_blocking=True)
        adj_matrix = b_data['adj_matrix'].view(
            self.bs * self.cfg.num_srl, 11, -1).cuda(non_blocking=True)

        verb_mask = b_data['verb_token_mask'].view(
            self.bs * self.cfg.num_srl,
            self.cfg.num_verbs, -1).cuda(non_blocking=True)
        noun_mask = b_data['noun_token_mask'].view(
            self.bs * self.cfg.num_srl,
            self.cfg.num_nouns, -1).cuda(non_blocking=True)

        root_node, vn_node = self.sr_encoder(sent_bert, sent_mask, verb_mask,
                                             noun_mask, node_role, adj_matrix)

        root_node = root_node.view(-1, self.cfg.num_srl, self.cfg.hid_dim)
        vn_node = vn_node.view(-1, self.cfg.num_srl, 10, self.cfg.hid_dim)
        node_role = node_role.view(-1, self.cfg.num_srl, 10)
        return root_node, vn_node, node_role


class RelationReasoning(nn.Module):
    def __init__(self, hidden_dim=512, num_head=4, dropout=0.):
        super(RelationReasoning, self).__init__()
        self.hid_dim = hidden_dim

        self.sr_que_attn = nn.MultiheadAttention(hidden_dim, num_head)
        self.or_que_attn = nn.MultiheadAttention(hidden_dim, num_head)

        self.img_que_attn = nn.MultiheadAttention(hidden_dim, num_head)
        self.sent_que_attn = nn.MultiheadAttention(hidden_dim, num_head)

        self.vis_sem_encoder = MultiGranularityVSEncoder(
            hidden_dim=hidden_dim,
            encoder_name='gcn',
            dropout=dropout,
        )

        self.que_att = PadLSTM(
            hidden_dim,
            hidden_dim // 2,
            num_layers=1,
            bidirectional=True,
            # dropout=dropout,
        )

    def forward(self, q_embed, q_mask, img_feat, obj_feat,
                sent_embed, role_embed, role_mask):
        """
        @param q_embed: [bs, n_token, 512]
        @param q_mask: [bs, n_token]
        @param img_feat: [bs, n_img, 512]
        @param obj_feat: [bs, n_img, n_obj, 512]
        @param sent_embed: [bs, n_know, 512]
        @param role_embed: [bs, n_know, 10, 512]
        @param role_mask: [bs, n_know, 10]
        @return:
        """
        q_embed = q_embed.permute(1, 0, 2).contiguous()  # [#n_token, bs, 512]

        # semantic role
        role_embed = role_embed.masked_fill(
            role_mask.unsqueeze(-1) == 0, 1e-10).sum(2)
        role_mask = (role_mask > 0).float().sum(-1)
        role_mask = role_mask.masked_fill(role_mask == 0, 10.).unsqueeze(-1)
        role_mask = role_mask.masked_fill(role_mask != role_mask, 10.)
        role_embed = role_embed / role_mask  # [bs, n_know, 512]

        # object relation
        obj_feat = obj_feat.mean(dim=2)  # [bs, n_img, 512]
        # obj_feat = obj_feat.max(dim=2)[0]
        
        # feature - question attention
        role_embed, _ = self.sr_que_attn(  # [n_know, bs, 512]
            role_embed.permute(1, 0, 2),
            q_embed, q_embed, key_padding_mask=~q_mask)

        sent_embed, _ = self.sent_que_attn(  # [n_know, bs, 512]
            sent_embed.permute(1, 0, 2),
            q_embed, q_embed, key_padding_mask=~q_mask)

        obj_feat, _ = self.or_que_attn(  # [#img, bs, 512]
            obj_feat.permute(1, 0, 2),
            q_embed, q_embed, key_padding_mask=~q_mask)

        img_feat, _ = self.img_que_attn(  # [#img, bs, 512]
            img_feat.permute(1, 0, 2),
            q_embed, q_embed, key_padding_mask=~q_mask)

        vis_sem_embed = self.vis_sem_encoder(  # [bs, 512]
            obj_feat.permute(1, 0, 2),
            img_feat.permute(1, 0, 2),
            role_embed.permute(1, 0, 2),
            sent_embed.permute(1, 0, 2),
        )

        _, (q_embed, _) = self.que_att(
            q_embed,
            q_mask.sum(1).cpu()
        )
        q_embed = torch.cat([q_embed[0], q_embed[1]], dim=-1)   # [bs, 512]

        multi_embeds = torch.cat([vis_sem_embed, q_embed], dim=-1)

        return multi_embeds


if __name__ == '__main__':
    import fire

    fire.Fire()