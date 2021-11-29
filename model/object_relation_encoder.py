# @File : object_relation_encoder.py 
# @Time : 2020/8/19 
# @Email : jingjingjiang2017@gmail.com

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.weight_norm import weight_norm

from module.adj_learner import AdjMatrixEdgeLearner
from module.gat import GATEncoder
from module.gcn import GCNEncoder
from module.fc import FCNet
from module.co_attention import CoAttention


class VisualRelationEncoder(nn.Module):
    def __init__(self, input_dim=2048, sp_dim=64, hidden_dim=512,
                 encoder_name='gcn', gcn_layer=1, num_head=1, dropout=0.,
                 gcn_attention=False, with_co_attn=False):
        super(VisualRelationEncoder, self).__init__()
        self.hid_dim = hidden_dim
        self.with_co_attn = with_co_attn

        self.attr_transform = nn.Sequential(
            # nn.LayerNorm(768),
            nn.BatchNorm1d(640),
            weight_norm(nn.Linear(768, hidden_dim), dim=None),
            nn.BatchNorm1d(640),  # 32 * 12
            # nn.LayerNorm(hidden_dim),
            nn.ReLU(True),
            nn.Dropout(p=dropout, inplace=True),
        )

        self.adj_leaner = AdjMatrixEdgeLearner(
            dim_input=hidden_dim,
            dim_hidden=hidden_dim
        )

        self.obj_transform = nn.Sequential(
            # nn.LayerNorm(input_dim),
            nn.BatchNorm1d(640),
            weight_norm(nn.Linear(input_dim, hidden_dim), dim=None),
            nn.BatchNorm1d(640),
            # nn.LayerNorm(hidden_dim),
            nn.ReLU(True),
            nn.Dropout(p=dropout, inplace=True),
        )

        self.sp_transform = nn.Sequential(
            # nn.LayerNorm(6),
            nn.BatchNorm1d(640),
            weight_norm(nn.Linear(6, sp_dim), dim=None),
            # nn.LayerNorm(sp_dim),
            nn.BatchNorm1d(640),
            nn.ReLU(True),
        )

        self.combine_transform = nn.Sequential(
            nn.LayerNorm(hidden_dim * 2),
            weight_norm(nn.Linear(hidden_dim * 2, hidden_dim), dim=None),
            # nn.BatchNorm1d(384),  # 32 * 12
            nn.LayerNorm(hidden_dim),
            nn.ReLU(True),
            nn.Dropout(p=dropout, inplace=True),
        )

        self.combine_transform2 = nn.Sequential(
            nn.LayerNorm(hidden_dim + sp_dim),
            weight_norm(nn.Linear(hidden_dim + sp_dim, hidden_dim), dim=None),
            # nn.BatchNorm1d(384),  # 32 * 12
            nn.LayerNorm(hidden_dim),
            nn.ReLU(True),
            nn.Dropout(p=dropout, inplace=True),
        )

        self.multi_attn = nn.MultiheadAttention(hidden_dim, 1)
        self.bias = FCNet([11, 1], '', 0, bias=True)

        self.factor = nn.Parameter(
            torch.FloatTensor([0.5, 0.5]), requires_grad=True)

        if encoder_name == 'gcn':
            self.encoder_a = GCNEncoder(
                dim_input=hidden_dim,
                dim_hidden=hidden_dim,
                num_hidden_layers=gcn_layer,
                dropout=dropout,
                attention=gcn_attention,
                embed_first=False
            )
            self.encoder_s = GCNEncoder(
                dim_input=hidden_dim + sp_dim,
                dim_hidden=hidden_dim,
                num_hidden_layers=gcn_layer,
                dropout=dropout,
                attention=gcn_attention,
                embed_first=False
            )
        elif encoder_name == 'gat':
            self.encoder_a = GATEncoder(
                hidden_dim,
                hidden_dim // num_head,
                dropout=dropout,
                n_head=num_head
            )
            self.encoder = GATEncoder(
                hidden_dim + sp_dim,
                hidden_dim // num_head,
                dropout=dropout,
                n_head=num_head
            )
        if self.with_co_attn:
            self.co_attn = CoAttention(hidden_dim)

    def forward(self, obj_feat, attr_feat, sp_feat, adj_matrix):
        """
        @param obj_feat: [bs, n_img, n_obj, 2048]
        @param attr_feat: [bs, n_img, n_obj, 768]
        @param adj_matrix: [bs, n_img, n_obj, n_obj, 11]
        @param sp_feat: [bs, n_img, n_obj, 6]
        @return: [bs, n_img, 512]
        """
        _, n_img, n_obj, _ = obj_feat.shape
        # [bs * n_img, n_obj, 512]
        obj_feat = self.obj_transform(obj_feat.view(-1, n_img * n_obj, 2048)
                                      ).view(-1, n_obj, self.hid_dim)

        # learning the semantic relation --------------------------------------
        # [bs * n_img, n_obj, 512]
        attr_feat = self.attr_transform(attr_feat.view(-1, n_img * n_obj, 768)
                                        ).view(-1, n_obj, self.hid_dim)
        attr_feat = self.combine_transform(torch.cat([obj_feat, attr_feat], -1))

        node_embed_a = self.encoder_a(attr_feat, self.adj_leaner(attr_feat))

        # learning the spatial relation --------------------------------------
        # [bs * n_img, n_obj, 64]
        sp_feat = self.sp_transform(
            sp_feat.view(-1, n_img * n_obj, 6)).view(-1, n_obj, 64)
        # [bs * n_img, n_obj, 512]
        obj_feat = self.combine_transform2(torch.cat([obj_feat, sp_feat], -1))

        obj_feat, obj_attn = self.multi_attn(  # [#obj, bs * n_img, 512]
            obj_feat.permute(1, 0, 2), obj_feat.permute(1, 0, 2),
            obj_feat.permute(1, 0, 2))

        adj_matrix = adj_matrix.view(-1, n_obj, n_obj, 11)
        bias = self.bias(adj_matrix).squeeze(-1)  # [bs * n_img, n_obj, n_obj]
        # [bs * n_img, n_obj, n_obj]
        adj_matrix = obj_attn.masked_fill(adj_matrix.sum(-1) == 0, -1e10) + bias
        adj_matrix = F.softmax(adj_matrix, dim=-1)
        node_embed = self.encoder_s(obj_feat.permute(1, 0, 2), adj_matrix)

        if self.with_co_attn:
            node_embed_a, node_embed = self.co_attn(node_embed_a, node_embed)

        # node_embed = self.alpha * node_embed_a + self.gamma * node_embed
        factor = F.softmax(self.factor, dim=0)
        node_embed = factor[0] * node_embed_a + factor[1] * node_embed

        return node_embed.view(-1, n_img, n_obj, self.hid_dim)


class ObjectRelationEncoder(nn.Module):
    def __init__(self, input_dim=2048, sp_dim=64, hidden_dim=512,
                 encoder_name='gcn', gcn_layer=1, num_head=1, dropout=0.,
                 gcn_attention=False, with_co_attn=False):
        super(ObjectRelationEncoder, self).__init__()
        self.hid_dim = hidden_dim
        self.with_co_attn = with_co_attn

        self.obj_transform = nn.Sequential(
            nn.LayerNorm(2048),
            weight_norm(nn.Linear(input_dim, hidden_dim), dim=None),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout, inplace=True),
        )

        self.sp_transform = nn.Sequential(
            nn.LayerNorm(6),
            weight_norm(nn.Linear(6, sp_dim), dim=None),
            nn.LayerNorm(sp_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout, inplace=True),
        )

        self.bias = FCNet([11, 1], '', 0, bias=True)
        self.multi_attn = nn.MultiheadAttention(hidden_dim + sp_dim,
                                                num_head)

        self.adj_leaner = AdjMatrixEdgeLearner(
            dim_input=hidden_dim,
            dim_hidden=hidden_dim,
            k=5,
            to_sparse=True
        )

        self.factor = nn.Parameter(
            torch.FloatTensor([0.5, 0.5]), requires_grad=True)

        if encoder_name == 'gcn':
            self.encoder_a = GCNEncoder(
                hidden_dim,
                hidden_dim,
                num_hidden_layers=gcn_layer,
                dropout=dropout,
                attention=gcn_attention,
                embed_first=False
            )
            self.encoder_s = GCNEncoder(
                hidden_dim + sp_dim,
                hidden_dim,
                num_hidden_layers=gcn_layer,
                dropout=dropout,
                attention=gcn_attention,
                embed_first=True
            )

        elif encoder_name == 'gat':
            self.encoder = GATEncoder(
                hidden_dim,
                hidden_dim // num_head,
                dropout=dropout,
                n_head=num_head
            )

        if self.with_co_attn:
            self.co_attn = CoAttention(hidden_dim)
            # self.encoder = GATEncoder(hidden_dim + sp_dim,
            #                           hidden_dim // num_head,
            #                           dropout=dropout, n_head=num_head)
        # self.pooling = nn.AdaptiveAvgPool1d(1)

    def forward(self, obj_feat, sp_feat, adj_matrix):
        """
        @param obj_feat: [bs, n_img, n_obj, 2048]
        @param adj_matrix: [bs, n_img, n_obj, n_obj, 11]
        @param sp_feat: [bs, n_img, n_obj, 6]
        @return: [bs, n_img, 512]
        """
        _, n_img, n_obj, _ = obj_feat.shape

        # [bs * n_img, n_obj, 512]
        obj_feat = self.obj_transform(obj_feat.view(-1, n_img * n_obj, 2048)
                                      ).view(-1, n_obj, self.hid_dim)

        node_embed_a = self.encoder_a(obj_feat, self.adj_leaner(obj_feat))

        # [bs * n_img, n_obj, 64]
        sp_feat = self.sp_transform(
            sp_feat.view(-1, n_img * n_obj, 6)).view(-1, n_obj, 64)
        obj_feat = torch.cat([obj_feat, sp_feat], dim=-1)

        obj_feat, obj_attn = self.multi_attn(  # [#obj, bs * n_img, 512]
            obj_feat.permute(1, 0, 2), obj_feat.permute(1, 0, 2),
            obj_feat.permute(1, 0, 2))

        adj_matrix = adj_matrix.view(-1, n_obj, n_obj, 11)
        bias = self.bias(adj_matrix).squeeze(-1)  # [bs * n_img, n_obj, n_obj]
        # [bs * n_img, n_obj, n_obj]
        adj_matrix = obj_attn.masked_fill(adj_matrix.sum(-1) == 0, -1e10) + bias
        adj_matrix = F.softmax(adj_matrix, dim=-1)
        node_embed = self.encoder_s(obj_feat.permute(1, 0, 2), adj_matrix)

        if self.with_co_attn:
            node_embed_a, node_embed = self.co_attn(node_embed_a, node_embed)

        # factor = F.softmax(self.factor, dim=0)
        # node_embed = factor[0] * node_embed_a + factor[1] * node_embed
        node_embed = node_embed_a + node_embed

        return node_embed.view(-1, n_img, n_obj, self.hid_dim)


class ObjectRelationFCEncoder(nn.Module):
    def __init__(self, input_dim=2048, sp_dim=64, hidden_dim=512,
                 encoder_name='gcn', gcn_layer=1, num_head=1, dropout=0.,
                 gcn_attention=False, with_co_attn=False):
        super(ObjectRelationFCEncoder, self).__init__()
        self.hid_dim = hidden_dim
        self.with_co_attn = with_co_attn

        self.obj_transform = nn.Sequential(
            nn.LayerNorm(2048),
            weight_norm(nn.Linear(input_dim, hidden_dim), dim=None),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout, inplace=True),
        )

        self.sp_transform = nn.Sequential(
            nn.LayerNorm(6),
            weight_norm(nn.Linear(6, sp_dim), dim=None),
            nn.LayerNorm(sp_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout, inplace=True),
        )
        
        self.fc_transform = nn.Sequential(
            nn.LayerNorm(hidden_dim + sp_dim),
            weight_norm(nn.Linear(hidden_dim + sp_dim, hidden_dim), dim=None),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout, inplace=True),
        )

    def forward(self, obj_feat, sp_feat, adj_matrix):
        """
        @param obj_feat: [bs, n_img, n_obj, 2048]
        @param adj_matrix: [bs, n_img, n_obj, n_obj, 11]
        @param sp_feat: [bs, n_img, n_obj, 6]
        @return: [bs, n_img, 512]
        """
        _, n_img, n_obj, _ = obj_feat.shape

        # [bs * n_img, n_obj, 512]
        obj_feat = self.obj_transform(obj_feat.view(-1, n_img * n_obj, 2048)
                                      ).view(-1, n_obj, self.hid_dim)

        # [bs * n_img, n_obj, 64]
        sp_feat = self.sp_transform(
            sp_feat.view(-1, n_img * n_obj, 6)).view(-1, n_obj, 64)
        
        # [bs * n_img, n_obj, 512 + 64]
        obj_feat = torch.cat([obj_feat, sp_feat], dim=-1)

        obj_feat = self.fc_transform(obj_feat)

        return obj_feat.view(-1, n_img, n_obj, self.hid_dim)


# class ObjectRelationEncoder(nn.Module):
#     def __init__(self, input_dim=2048, sp_dim=64, hidden_dim=512,
#                  encoder_name='gcn', gcn_layer=1, num_head=1, dropout=0.,
#                  gcn_attention=False):
#         super(ObjectRelationEncoder, self).__init__()
#         self.hid_dim = hidden_dim
#
#         # self.obj_transform = FCNet([input_dim, hidden_dim], dropout=dropout)
#         # self.sp_transform = FCNet([7, 64])
#         self.obj_transform = nn.Sequential(
#             weight_norm(nn.Linear(input_dim, hidden_dim), dim=None),
#             # nn.LayerNorm(hidden_dim),
#             nn.BatchNorm1d(384),  # 32 * 12
#             nn.ReLU(True))
#         self.sp_transform = nn.Sequential(
#             weight_norm(nn.Linear(6, sp_dim), dim=None),
#             # nn.LayerNorm(sp_dim),
#             nn.BatchNorm1d(384),
#             nn.ReLU(True))
#
#         self.bias = FCNet([11, 1], '', 0, bias=True)
#         self.multi_attn = nn.MultiheadAttention(hidden_dim, num_head)
#
#         # self.adj_leaner = AdjMatrixEdgeLearner(dim_input=hidden_dim,
#         #                                        dim_hidden=hidden_dim // 4)
#
#         if encoder_name == 'gcn':
#             self.encoder = GCNEncoder(hidden_dim, hidden_dim,
#                                       num_hidden_layers=gcn_layer,
#                                       dropout=dropout,
#                                       attention=gcn_attention,
#                                       embed_first=False)
#         elif encoder_name == 'gat':
#             self.encoder = GATEncoder(hidden_dim,
#                                       hidden_dim // num_head,
#                                       dropout=dropout, n_head=num_head)
#
#             # self.encoder = GATEncoder(hidden_dim + sp_dim,
#             #                           hidden_dim // num_head,
#             #                           dropout=dropout, n_head=num_head)
#         # self.pooling = nn.AdaptiveAvgPool1d(1)
#
#     def forward(self, obj_feat, sp_feat, adj_matrix):
#         """
#         @param obj_feat: [bs, n_img, n_obj, 2048]
#         @param adj_matrix: [bs, n_img, n_obj, n_obj, 11]
#         @param sp_feat: [bs, n_img, n_obj, 6]
#         @return: [bs, n_img, 512]
#         """
#         _, n_img, n_obj, _ = obj_feat.shape
#
#         # [bs * n_img, n_obj, 512]
#         obj_feat = self.obj_transform(obj_feat.view(-1, n_img * n_obj, 2048)
#                                       ).view(-1, n_obj, self.hid_dim)
#         # [bs * n_img, n_obj, 64]
#         # sp_feat = self.sp_transform(
#         #     sp_feat.view(-1, n_img * n_obj, 6)).view(-1, n_obj, 64)
#
#         obj_feat, obj_attn = self.multi_attn(  # [#obj, bs * n_img, 512]
#             obj_feat.permute(1, 0, 2), obj_feat.permute(1, 0, 2),
#             obj_feat.permute(1, 0, 2))
#
#         adj_matrix = adj_matrix.view(-1, n_obj, n_obj, 11)
#         bias = self.bias(adj_matrix).squeeze(-1)  # [bs * n_img, n_obj, n_obj]
#         # [bs * n_img, n_obj, n_obj]
#         adj_matrix = obj_attn.masked_fill(adj_matrix.sum(-1) == 0, -1e10) + bias
#         adj_matrix = F.softmax(adj_matrix, dim=-1)
#         node_embed = self.encoder(obj_feat.permute(1, 0, 2), adj_matrix)
#
#         return node_embed.view(-1, n_img, n_obj, self.hid_dim)


# class ObjectRelationEncoder(nn.Module):
#     def __init__(self, input_dim=2048, sp_dim=64, hidden_dim=512,
#                  encoder_name='gcn', gcn_layer=1, num_head=1, dropout=0.,
#                  gcn_attention=False):
#         super(ObjectRelationEncoder, self).__init__()
#         self.hid_dim = hidden_dim
#
#         # self.obj_transform = FCNet([input_dim, hidden_dim], dropout=dropout)
#         # self.sp_transform = FCNet([7, 64])
#         self.obj_transform = nn.Sequential(
#             weight_norm(nn.Linear(input_dim, hidden_dim), dim=None),
#             # nn.LayerNorm(hidden_dim),
#             nn.BatchNorm1d(640),
#             nn.ReLU(True))
#         self.sp_transform = nn.Sequential(
#             weight_norm(nn.Linear(6, sp_dim), dim=None),
#             # nn.LayerNorm(sp_dim),
#             nn.BatchNorm1d(640),
#             nn.ReLU(True))
#
#         self.adj_leaner = AdjMatrixEdgeLearner(dim_input=hidden_dim,
#                                                dim_hidden=hidden_dim // 4)
#
#         if encoder_name == 'gcn':
#             self.encoder_l = GCNEncoder(hidden_dim, hidden_dim,
#                                         num_hidden_layers=gcn_layer,
#                                         dropout=dropout,
#                                         attention=gcn_attention,
#                                         embed_first=False)
#             self.encoder = GCNEncoder(hidden_dim + sp_dim, hidden_dim,
#                                       num_hidden_layers=gcn_layer,
#                                       dropout=dropout,
#                                       attention=gcn_attention,
#                                       embed_first=True)
#         elif encoder_name == 'gat':
#             self.encoder_l = GATEncoder(hidden_dim,
#                                         hidden_dim // num_head,
#                                         dropout=dropout, n_head=num_head)
#
#             self.encoder = GATEncoder(hidden_dim + sp_dim,
#                                       hidden_dim // num_head,
#                                       dropout=dropout, n_head=num_head)
#         # self.pooling = nn.AdaptiveAvgPool1d(1)
#
#     def forward(self, obj_feat, sp_feat, adj_matrix):
#         """
#         @param obj_feat: [bs, n_img, n_obj, 2048]
#         @param adj_matrix: [bs, n_img, n_obj, n_obj]
#         @param sp_feat: [bs, n_img, n_obj, 6]
#         @return: [bs, n_img, 512]
#         """
#         _, n_img, n_obj, _ = obj_feat.shape
#
#         # [bs * n_img, n_obj, 512]
#         obj_feat = self.obj_transform(
#             obj_feat.view(-1, n_img * n_obj, 2048)).view(-1, n_obj, self.hid_dim)
#         node_embed_l = self.encoder_l(obj_feat, self.adj_leaner(obj_feat))
#
#         # [bs * n_img, n_obj, 64]
#         sp_feat = self.sp_transform(
#             sp_feat.view(-1, n_img * n_obj, 6)).view(-1, n_obj, 64)
#         adj_matrix = adj_matrix.view(-1, n_obj, n_obj)
#         # node_embed = self.encoder(torch.cat([obj_feat, sp_feat], dim=-1),
#         #                           adj_matrix.masked_fill_(adj_matrix < 0.1, 0.))
#         node_embed = self.encoder(
#             torch.cat([obj_feat, sp_feat], dim=-1), adj_matrix)
#
#         node_embed = 0.5 * node_embed_l + 0.5 * node_embed
#         return node_embed.view(-1, n_img, n_obj, self.hid_dim)
