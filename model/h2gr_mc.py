# @File  :h2gr_mc.py
# @Time  :2020/9/28
# @Desc  :
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict
from torch.nn.utils.weight_norm import weight_norm

from module.dynamic_rnn import PadLSTM
from model.classifier import SimpleClassifier
from model.object_relation_encoder import ObjectRelationEncoder
from model.object_relation_encoder import VisualRelationEncoder
from model.semantic_role_encoder import SemanticRoleEncoder
from model.visual_semantic_encoder import VisualSemanticEncoder
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

        self.a_encoding = nn.Sequential(
            nn.LayerNorm(cfg.token_dim),
            weight_norm(nn.Linear(cfg.token_dim, cfg.hid_dim), dim=None),
            nn.LayerNorm(cfg.hid_dim),
            nn.ReLU(),
            nn.Dropout(p=cfg.dropout, inplace=True),
        )

        # knowledge encoder
        self.sr_encoder = SemanticRoleEncoder(
            input_dim=cfg.token_dim,
            hidden_dim=cfg.hid_dim,
            encoder_name='gcn',
            gcn_attention=False,
            dropout=cfg.dropout,
            with_root_node=self.with_root_node
        )

        # image encoder
        self.img_encoder = nn.Sequential(
            nn.LayerNorm(2048),
            weight_norm(nn.Linear(2048, cfg.hid_dim), dim=None),
            nn.LayerNorm(cfg.hid_dim),
            nn.ReLU(),
            nn.Dropout(p=cfg.dropout, inplace=True),
        )
        # nn.init.kaiming_normal_(self.img_encoder[1].weight)

        # object relation encoder
        self.or_encoder = ObjectRelationEncoder(
            hidden_dim=cfg.hid_dim,
            encoder_name='gcn',
            gcn_attention=True,
            num_head=cfg.num_head,
            dropout=cfg.dropout,
            with_co_attn=False
        )
        # self.or_encoder = VisualRelationEncoder(
        #     hidden_dim=cfg.hid_dim,
        #     encoder_name='gcn',
        #     gcn_attention=True,
        #     dropout=cfg.dropout,
        #     num_head=cfg.num_head,
        #     with_co_attn=True
        # )

        self.rel_reasoning = RelationReasoning(
            hidden_dim=cfg.hid_dim,
            num_head=cfg.num_head,
            dropout=cfg.dropout,
        )

        # answer prediction
        # self.classifier = SimpleClassifier(
        #     cfg.hid_dim * 3,
        #     cfg.hid_dim,
        #     1,
        #     dropout=cfg.dropout
        # )

    def forward(self, batch_data) -> Dict:
        target = batch_data.get('target').long().cuda(non_blocking=True)
        self.bs = target.shape[0]
        # ########################## feature encoder ###########################
        q_embed = self.q_encoding(  # [bs, n_token, 512]
            batch_data['q_bert'].float().cuda(non_blocking=True))
        # qa_embed = qa_embed.view(self.bs, 5, -1, self.cfg.hid_dim)
        # qa_mask = batch_data['qa_mask'].bool().cuda().view(self.bs * 5, -1)
        a_embed = self.a_encoding(  # [bs, 5, 512]
            batch_data['a_bert'].float().cuda(non_blocking=True))

        # semantic role encoder
        sent_embed, role_embed, role_mask = self.forward_semantic_role(
            batch_data)

        # image encoder
        img_embed = self.img_encoder(  # [bs, #img, 512]
            batch_data.get('img_feat').float().cuda(non_blocking=True))

        # object relation encoder
        obj_embed = self.or_encoder(  # [bs, #img, #obj, 512]
            batch_data.get('obj_feat').float().cuda(non_blocking=True),
            batch_data.get('sp_feat').float().cuda(non_blocking=True),
            batch_data.get('obj_adj_matrix').float().cuda(non_blocking=True)
        )

        # ######################### relation reasoning ##########################
        logit = self.rel_reasoning(
            q_embed,
            # [bs, 4, n_token]
            batch_data['q_mask'].bool().cuda(non_blocking=True),
            a_embed,
            img_embed,
            obj_embed,
            sent_embed, role_embed, role_mask
        )

        # logit = self.classifier(fusion_embed)

        return {'target': target,  # [bs]
                'logit': logit.squeeze()
                }

    def forward_semantic_role(self, b_data):
        sent_bert = b_data['sent_bert'].float().view(
            self.bs * self.cfg.num_srl, -1, 768).cuda(non_blocking=True)
        sent_mask = b_data['sent_mask'].view(
            self.bs * self.cfg.num_srl, -1).cuda(non_blocking=True)
        node_role = b_data['node_role'].view(
            self.bs * self.cfg.num_srl, -1).long().cuda(non_blocking=True)
        adj_matrix = b_data['adj_matrix'].float().view(
            self.bs * self.cfg.num_srl, 11, -1).cuda(non_blocking=True)
        verb_mask = b_data['verb_token_mask'].view(
            self.bs * self.cfg.num_srl, self.cfg.num_verbs, -1).cuda(
            non_blocking=True)
        noun_mask = b_data['noun_token_mask'].view(
            self.bs * self.cfg.num_srl, self.cfg.num_nouns, -1).cuda(
            non_blocking=True)

        root_node, vn_node = self.sr_encoder(sent_bert, sent_mask, verb_mask,
                                             noun_mask, node_role, adj_matrix)

        root_node = root_node.view(-1, self.cfg.num_srl, self.cfg.hid_dim)
        vn_node = vn_node.view(-1, self.cfg.num_srl, 10, self.cfg.hid_dim)
        node_role = node_role.view(-1, self.cfg.num_srl, 10)
        return root_node, vn_node, node_role


class RelationReasoning(nn.Module):
    def __init__(self, hidden_dim=512, num_head=4, dropout=0.5):
        super(RelationReasoning, self).__init__()
        self.hid_dim = hidden_dim

        self.sr_que_attn = nn.MultiheadAttention(hidden_dim, num_head)
        self.or_que_attn = nn.MultiheadAttention(hidden_dim, num_head)

        self.img_que_attn = nn.MultiheadAttention(hidden_dim, num_head)
        self.sent_que_attn = nn.MultiheadAttention(hidden_dim, num_head)

        # self.vis_sem_encoder = VisualSemanticEncoder(
        #     hidden_dim=hidden_dim,
        #     encoder_name='gcn',
        #     dropout=dropout,
        # )
        self.vis_sem_encoder = MultiGranularityVSEncoder(
            hidden_dim=hidden_dim,
            encoder_name='gcn',
            gcn_attention=True,
            dropout=dropout,
        )
        #
        # self.or_lstm = nn.LSTM(
        #     hidden_dim,
        #     hidden_dim // 2,
        #     num_layers=1,
        #     batch_first=False,
        #     bidirectional=True
        # )
        #
        # self.img_lstm = nn.LSTM(
        #     hidden_dim,
        #     hidden_dim // 2,
        #     num_layers=1,
        #     batch_first=False,
        #     bidirectional=True
        # )
        #
        # self.sr_lstm = nn.LSTM(
        #     hidden_dim,
        #     hidden_dim // 2,
        #     num_layers=1,
        #     batch_first=False,
        #     bidirectional=True
        # )
        #
        # self.sent_lstm = nn.LSTM(
        #     hidden_dim,
        #     hidden_dim // 2,
        #     num_layers=1,
        #     batch_first=False,
        #     bidirectional=True
        # )
        # init_rnn_weights(self.or_lstm, rnn_type='lstm')
        # init_rnn_weights(self.img_lstm, rnn_type='lstm')
        # init_rnn_weights(self.sr_lstm, rnn_type='lstm')
        # init_rnn_weights(self.sent_lstm, rnn_type='lstm')
        #
        self.que_att = PadLSTM(
            hidden_dim,
            hidden_dim // 2,
            num_layers=1,
            bidirectional=True,
            # dropout=dropout,
        )

        # self.ada_attn = weight_norm(nn.Linear(hidden_dim, 1), dim=None)

        # self.img_que_co = CoAttention(hidden_dim=hidden_dim, sum=True)
        # self.sent_que_co = CoAttention(hidden_dim=hidden_dim, sum=True)
        self.classifier = SimpleClassifier(
            self.hid_dim * 3,
            self.hid_dim,
            1,
            dropout=dropout
        )

    def forward(self, q_embed, q_mask, a_embed,
                img_feat, obj_feat,
                sent_embed, role_embed, role_mask):
        """
        @param q_embed: [bs,  n_token, 512]
        @param q_mask: [bs, n_token]
        @param img_feat: [bs, n_img, 512]
        @param obj_feat: [bs, n_img, n_obj, 512]
        @param sent_embed: [bs, n_know, 512]
        @param role_embed: [bs, n_know, 10, 512]
        @param role_mask: [bs, n_know, 10]
        @return:
        """
        self.bs = q_embed.shape[0]
        q_embed = q_embed.permute(1, 0, 2).contiguous()  # [#n_token, bs, 512]

        # semantic role
        role_embed = role_embed.masked_fill(
            role_mask.unsqueeze(-1) == 0, 0.).sum(2)
        role_mask = (role_mask > 0).float().sum(-1)
        role_mask = role_mask.masked_fill(role_mask == 0, 10).unsqueeze(-1)
        role_embed = role_embed / role_mask  # [bs, n_know, 512]

        # object relation
        obj_feat = obj_feat.mean(2)  # [bs, n_img, 512]

        # feature-question attention
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

        vis_sem_embed = self.vis_sem_encoder(  # [bs, x, 512]
            obj_feat.permute(1, 0, 2),
            img_feat.permute(1, 0, 2),
            role_embed.permute(1, 0, 2),
            sent_embed.permute(1, 0, 2),
        )

        # self.or_lstm.flatten_parameters()
        # _, (obj_feat, _) = self.or_lstm(obj_feat)
        # obj_feat = torch.cat([obj_feat[0], obj_feat[1]], dim=-1)
        # # obj_feat = obj_feat.squeeze()  # [bs, 512]
        #
        # self.img_lstm.flatten_parameters()
        # _, (img_feat, _) = self.img_lstm(img_feat)
        # img_feat = torch.cat([img_feat[0], img_feat[1]], dim=-1)
        # # img_feat = img_feat.squeeze()
        #
        # self.sr_lstm.flatten_parameters()
        # _, (role_embed, _) = self.sr_lstm(role_embed)
        # role_embed = torch.cat([role_embed[0], role_embed[1]], dim=-1)
        # # role_embed = role_embed.squeeze()  # [bs, 512]
        #
        # self.sent_lstm.flatten_parameters()
        # _, (sent_embed, _) = self.sent_lstm(sent_embed)
        # sent_embed = torch.cat([sent_embed[0], sent_embed[1]], dim=-1)
        # sent_embed = sent_embed.squeeze()

        _, (q_embed, _) = self.que_att(q_embed, q_mask.sum(1))
        q_embed = torch.cat([q_embed[0], q_embed[1]], dim=-1)  # [bs, 512]

        # [bs, x, 512]
        # multi_embeds = torch.stack([
        #     img_feat,
        #     sent_embed,
        #     obj_feat,
        #     role_embed,
        #     vis_sem_embed,
        #     # q_embed
        # ], dim=-2)

        # select feature   [bs, x, 512] -> [bs, 512]
        # attn = self.ada_attn(multi_embeds)
        # attn = F.softmax(attn, dim=1)
        # multi_embeds = torch.sum(multi_embeds * attn, dim=1)

        # answer - feature
        # attn = torch.einsum('bnd,bd->bn', multi_embeds, q_embed)
        # attn = F.softmax(attn, dim=1)
        # multi_embeds = torch.sum(multi_embeds * attn.unsqueeze(-1), dim=-2)

        # multi_embeds = multi_embeds.unsqueeze(1).repeat(1, 5, 1)
        score_answers = [self.classifier(torch.cat([
            vis_sem_embed,
            q_embed,
            a_embed[:, i, :]
        ], dim=-1)) for i in range(5)]

        return torch.cat(score_answers, dim=1)
        # return torch.cat([vis_sem_embed, q_embed, a_embed], dim=-1)


if __name__ == '__main__':
    import fire

    fire.Fire()
