# @File : semantic_role_encoder.py
# @Time : 2020/8/15
# @Email : jingjingjiang2017@gmail.com
import torch
import torch.nn as nn
from torch.nn.utils import weight_norm

from module.dynamic_rnn import DynamicRNN, PadLSTM
from module.gat import GATEncoder
from module.gcn import GCNEncoder, GPool


class SemanticRoleEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim=512, num_role=16, gcn_layer=1,
                 encoder_name='gcn', with_root_node=False, n_head=1,
                 dropout=0., gcn_attention=False, pooling=False):
        super(SemanticRoleEncoder, self).__init__()
        # num_role -> role category: 16
        self.max_token_in_sent = 30
        self.encoder_name = encoder_name
        # self.pooling = pooling
        self.with_root_node = with_root_node

        # 16 * 512
        self.role_embedding = nn.Embedding(num_role, hidden_dim)
        nn.init.uniform_(self.role_embedding.weight, -1.732, 1.732)

        if self.with_root_node:
            self.sent_embedding = nn.Sequential(
                nn.LayerNorm(input_dim),
                weight_norm(nn.Linear(input_dim, hidden_dim), dim=None),
                nn.LayerNorm(hidden_dim),
                # nn.ReLU(True),
                nn.ReLU(),
                nn.Dropout(p=dropout, inplace=True),
            )

            self.sent_encoder = PadLSTM(
                hidden_dim,
                hidden_dim // 2,
                num_layers=1,
                bidirectional=True,
            )
        else:
            self.sent_encoder = DynamicRNN(
                nn.LSTM(input_dim, hidden_dim, 1,
                        batch_first=True, bidirectional=True))

        # GCN
        if self.encoder_name == 'gcn':
            self.encoder = GCNEncoder(hidden_dim, hidden_dim,
                                      num_hidden_layers=gcn_layer,
                                      dropout=dropout,
                                      attention=gcn_attention,
                                      embed_first=False)
        elif self.encoder_name == 'gat':
            self.encoder = GATEncoder(input_dim, hidden_dim, n_head=n_head)

    def pool_phrases(self, sent_embed, phrase_mask, pool_type='avg'):
        if sent_embed.shape[1] > self.max_token_in_sent:
            sent_embed = sent_embed[:, :self.max_token_in_sent, :]
        elif sent_embed.shape[1] < self.max_token_in_sent:
            phrase_mask = phrase_mask[:, :, :sent_embed.shape[1]]

        if pool_type == 'avg':
            # (bs, num_phrases, max_sent_len, embed_size)
            phrase_mask = phrase_mask.float()
            phrase_embed = torch.bmm(phrase_mask, sent_embed) / torch.sum(
                phrase_mask, dim=2, keepdim=True).clamp_min(min=1)
            return phrase_embed
        elif pool_type == 'max':
            sent_embed = sent_embed.unsqueeze(1).masked_fill(
                phrase_mask.unsqueeze(3) == 0, -1e10)
            return torch.max(sent_embed, 2)[0]
        else:
            raise NotImplementedError

    def forward(self, sent_embed, sent_mask, verb_mask, noun_mask,
                node_role, adj_matrix):
        """
        @param sent_embed: (bs * num_srl, max_token_len, 768)
        @param sent_mask: (bs * num_srl, max_token_len)
        @param verb_mask: (bs * num_srl, num_verbs, max_token_len)
        @param noun_mask: (bs * num_srl, num_nouns, max_token_len)
        @param node_role: (bs * num_srl, num_verbs + num_nouns)
        @param adj_matrix: (bs * num_srl, 11, 11)
        @return:
        """
        # global sentence embeds : [bs, 512]
        # _, (root_embed, _) = self.sent_encoder(sent_embed, sent_mask.sum(dim=1))
        sent_embed = self.sent_embedding(sent_embed)   # [bs, x, 512]
        _, (root_embed, _) = self.sent_encoder(
            sent_embed.permute(1, 0, 2),
            sent_mask.sum(dim=1).cpu()
        )
        root_embed = torch.cat([root_embed[0], root_embed[1]], dim=-1)

        # (bs, num_phrase, embed_size)
        num_verb = verb_mask.size(1)
        v_embed = self.pool_phrases(sent_embed, verb_mask, pool_type='max')
        n_embed = self.pool_phrases(sent_embed, noun_mask, pool_type='max')

        # v_embed = v_embed * self.role_embedding(node_role[:, :num_verb])
        # n_embed = n_embed * self.role_embedding(node_role[:, num_verb:])
        v_embed = v_embed + v_embed * self.role_embedding(
            node_role[:, :num_verb])
        n_embed = n_embed + n_embed * self.role_embedding(
            node_role[:, num_verb:])

        # semantic role encoder
        if self.with_root_node:  # [bs*num_srg, 11, 768]
            node_embed = torch.cat([root_embed.unsqueeze(1),
                                    v_embed, n_embed
                                    ], 1)
            node_embed = self.encoder(node_embed, adj_matrix)
            return node_embed[:, 0, :], node_embed[:, 1:, :]
        else:
            node_embed = torch.cat(  # [bs*num_srg, 10, 768]
                [v_embed, n_embed], 1)
            node_embed = self.encoder(node_embed, adj_matrix[:, 1:, 1:])
            return root_embed, node_embed


class SemanticRoleGCNEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim=512, num_role=16,
                 gcn_layer=1,
                 dropout=0.5, gcn_attention=False, gcn_pooling=False):
        super(SemanticRoleGCNEncoder, self).__init__()
        # self.num_roles = num_role  # role category: 16
        self.max_token_in_sent = 30
        self.gcn_pooling = gcn_pooling

        self.sent_encoder = DynamicRNN(  # if layer>1: dropout=0.1
            nn.LSTM(input_dim, input_dim, 1,
                    batch_first=True, bidirectional=True))

        # ?????? 16 * 512
        self.role_embedding = nn.Embedding(num_role, input_dim)

        # GCN parameters
        self.gcn = GCNEncoder(input_dim, hidden_dim,
                              num_hidden_layers=gcn_layer,
                              dropout=dropout,
                              attention=gcn_attention,
                              embed_first=True)
        if self.gcn_pooling:
            self.graph_pooling = GPool(k=1, in_dim=hidden_dim)

    def pool_phrases(self, sent_embed, phrase_mask, pool_type='avg'):
        if sent_embed.shape[1] > self.max_token_in_sent:
            sent_embed = sent_embed[:, :self.max_token_in_sent, :]
        elif sent_embed.shape[1] < self.max_token_in_sent:
            phrase_mask = phrase_mask[:, :, :sent_embed.shape[1]]

        if pool_type == 'avg':
            # (bs, num_phrases, max_sent_len, embed_size)
            phrase_mask = phrase_mask.float()
            phrase_embed = torch.bmm(phrase_mask, sent_embed) / torch.sum(
                phrase_mask, dim=2, keepdim=True).clamp(min=1)
            return phrase_embed
        elif pool_type == 'max':
            sent_embed = sent_embed.unsqueeze(1).masked_fill(
                phrase_mask.unsqueeze(3) == 0, -1e10)
            return torch.max(sent_embed, 2)[0]
        else:
            raise NotImplementedError

    def forward(self, sent_embed, sent_mask, verb_mask, noun_mask,
                node_role, adj_matrix):
        """
        @param sent_embed: (bs * num_srl, max_token_len, 768)
        @param sent_mask: (bs * num_srl, max_token_len, 768)
        @param verb_mask: (bs * num_srl, num_verbs, max_token_len)
        @param noun_mask: (bs * num_srl, num_nouns, max_token_len)
        @param node_role: (bs * num_srl, num_verbs + num_nouns)
        @param adj_matrix: (bs * num_srl, 11, 11)
        @return:
        """
        # global sentence embeds : [bs, 512]
        # root_embed = sent_embed.sum(dim=1) / sent_mask.unsqueeze(dim=-1).sum(
        #     dim=1)
        _, (root_embed, _) = self.sent_encoder(sent_embed, sent_mask.sum(dim=1))
        # (batch, num_phrase, embed_size)
        num_verb = verb_mask.size(1)
        verb_embed = self.pool_phrases(sent_embed, verb_mask, pool_type='max')
        noun_embed = self.pool_phrases(sent_embed, noun_mask, pool_type='max')

        verb_embed = verb_embed * self.role_embedding(node_role[:, :num_verb])
        noun_embed = noun_embed * self.role_embedding(node_role[:, num_verb:])

        # gcn for srg encoding
        node_embed = torch.cat(  # [num_srg, 11, 512]
            [root_embed.unsqueeze(1), verb_embed, noun_embed], 1)
        node_ctx_embed = self.gcn(node_embed, adj_matrix)

        if self.gcn_pooling:
            global_sr_embed = self.graph_pooling(node_ctx_embed[:, 1:, :],
                                                 adj_matrix[:, 1:, 1:])
            return node_ctx_embed[:, 0, :], global_sr_embed
        else:
            return node_ctx_embed


class SemanticRoleGATEncoder(nn.Module):
    def __init__(self, input_dim=768, hidden_dim=512, num_role=16, dropout=0.5):
        super(SemanticRoleGATEncoder, self).__init__()
        # self.num_roles = num_role  # role category: 16
        self.max_token_in_sent = 30
        self.sent_encoder = DynamicRNN(  # if layer>1: dropout=0.1
            nn.LSTM(input_dim, input_dim, 1,
                    batch_first=True, bidirectional=True))

        # 16 * 512
        self.role_embedding = nn.Embedding(num_role, input_dim)

        self.gat = GATEncoder(input_dim, hidden_dim, n_head=1)

    def pool_phrases(self, sent_embed, phrase_mask, pool_type='avg'):
        if sent_embed.shape[1] > self.max_token_in_sent:
            sent_embed = sent_embed[:, :self.max_token_in_sent, :]
        elif sent_embed.shape[1] < self.max_token_in_sent:
            phrase_mask = phrase_mask[:, :, :sent_embed.shape[1]]

        if pool_type == 'avg':
            # (bs, num_phrases, max_sent_len, embed_size)
            phrase_mask = phrase_mask.float()
            phrase_embed = torch.bmm(phrase_mask, sent_embed) / torch.sum(
                phrase_mask, dim=2, keepdim=True).clamp(min=1)
            return phrase_embed
        elif pool_type == 'max':
            sent_embed = sent_embed.unsqueeze(1).masked_fill(
                phrase_mask.unsqueeze(3) == 0, -1e10)
            return torch.max(sent_embed, 2)[0]
        else:
            raise NotImplementedError

    def forward(self, sent_embed, sent_mask, verb_mask, noun_mask,
                node_role, adj_matrix):
        """
        @param sent_embed: (bs * num_srl, max_token_len, 768)
        @param sent_mask: (bs * num_srl, max_token_len)
        @param verb_mask: (bs * num_srl, num_verbs, max_token_len)
        @param noun_mask: (bs * num_srl, num_nouns, max_token_len)
        @param node_role: (bs * num_srl, num_verbs + num_nouns)
        @param adj_matrix: (bs * num_srl, 11, 11)
        @return:
        """
        # global sentence embeds : [bs, 512]
        # root_embed = sent_embed.sum(dim=1) / sent_mask.unsqueeze(dim=-1).sum(
        #     dim=1)
        _, (root_embed, _) = self.sent_encoder(sent_embed, sent_mask.sum(dim=1))
        # (batch, num_phrase, embed_size)
        num_verb = verb_mask.size(1)
        verb_embed = self.pool_phrases(sent_embed, verb_mask, pool_type='max')
        noun_embed = self.pool_phrases(sent_embed, noun_mask, pool_type='max')

        verb_embed = verb_embed * self.role_embedding(node_role[:, :num_verb])
        noun_embed = noun_embed * self.role_embedding(node_role[:, num_verb:])

        # gat for srg encoding
        node_embed = torch.cat(  # [bs*num_srg, 11, 768]
            [root_embed.unsqueeze(1), verb_embed, noun_embed], 1)
        node_ctx_embed = self.gat(node_embed, adj_matrix)
        return node_ctx_embed
        # node_embed = torch.cat(  # [bs*num_srg, 10, 768]
        #     [verb_embed, noun_embed], 1)
        # node_ctx_embed = self.gat(node_embed, adj_matrix[:, 1:, 1:])
        # return root_embed, node_ctx_embed
