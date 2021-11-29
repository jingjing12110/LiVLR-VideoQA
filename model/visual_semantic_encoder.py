# @File: visual_semantic_encoder.py
# @Time: 2020/8/14
# @Email: jingjingjiang2017@gmail.com
import torch
import torch.nn as nn
# from block import fusions

from module.gat import GATEncoder
from module.gcn import GCNEncoder
from module.co_attention import CoAttention
from module.adj_learner import AdjMatrixEdgeLearner


class VisualSemanticEncoder(nn.Module):
    def __init__(self, hidden_dim=512, encoder_name='gcn', gcn_layer=1,
                 num_head=4, gcn_attention=False, dropout=0.):
        super(VisualSemanticEncoder, self).__init__()
        self.hid_dim = hidden_dim
        # self.co_attn = CoAttention(hidden_dim)

        self.adj_leaner = AdjMatrixEdgeLearner(dim_input=hidden_dim,
                                               dim_hidden=hidden_dim // 4)

        if encoder_name == 'gcn':
            self.encoder = GCNEncoder(hidden_dim, hidden_dim,
                                      num_hidden_layers=gcn_layer,
                                      dropout=dropout,
                                      attention=gcn_attention,
                                      embed_first=False)
        elif encoder_name == 'gat':
            self.encoder = GATEncoder(hidden_dim, hidden_dim // num_head,
                                      dropout=dropout, n_head=num_head)

    def forward(self, vis_embed, sem_embed):
        """
        @param vis_embed: [bs, n_img, 512]
        @param sem_embed: [bs, n_know, 512]
        @return:
        """
        # vis_embed, sem_embed = self.co_attn(vis_embed, sem_embed)
        joint_embed = torch.cat([vis_embed, sem_embed], dim=1)
        joint_embed = self.encoder(joint_embed, self.adj_leaner(joint_embed))
        return joint_embed.mean(1)
        # return joint_embed.max(dim=1)[0]


class MultiGranularityVSEncoder(nn.Module):
    def __init__(self, hidden_dim=512, encoder_name='gcn', gcn_layer=1,
                 num_head=4, gcn_attention=False, dropout=0.,
                 ):
        super(MultiGranularityVSEncoder, self).__init__()
        self.hid_dim = hidden_dim
        # self.co_attn = CoAttention(hidden_dim)

        self.node_embedding = nn.Embedding(4, hidden_dim)
        # nn.init.uniform_(self.node_embedding.weight, -1.732, 1.732)

        self.adj_leaner = AdjMatrixEdgeLearner(
            dim_input=hidden_dim,
            dim_hidden=hidden_dim,
        )

        if encoder_name == 'gcn':
            self.encoder = GCNEncoder(hidden_dim, hidden_dim,
                                      num_hidden_layers=gcn_layer,
                                      dropout=dropout,
                                      attention=gcn_attention,
                                      embed_first=False)
        elif encoder_name == 'gat':
            self.encoder = GATEncoder(hidden_dim, hidden_dim // num_head,
                                      dropout=dropout, n_head=num_head)

    def forward(self, vis_obj_embed, vis_img_embed,
                sem_role_embed, sem_sent_embed):
        """
        @param vis_obj_embed: [bs, n_img, D]
        @param vis_img_embed: [bs, n_img, D]
        @param sem_role_embed: [bs, n_know, D]
        @param sem_sent_embed: [bs, n_know, D]
        @return:
        """
        n_img, n_know = vis_obj_embed.shape[1], sem_role_embed.shape[1]
        joint_embed = torch.cat([vis_obj_embed, vis_img_embed,
                                 sem_role_embed, sem_sent_embed], dim=1)
        # [bs, 2*n_img + 2*n_know]
        node_type = torch.zeros(joint_embed.shape[:-1]).long().cuda(
            non_blocking=True)
        node_type[:, n_img:2 * n_img] = 1
        node_type[:, 2 * n_img: 2 * n_img + n_know] = 2
        node_type[:, 2 * n_img + n_know:] = 3

        node_embed = self.node_embedding(node_type)
        joint_embed = joint_embed + joint_embed * node_embed

        joint_embed = self.encoder(joint_embed, self.adj_leaner(joint_embed))
        # return joint_embed.max(1)[0]
        return joint_embed.mean(1)


class MultiGranularityVSEncoderV2(nn.Module):
    def __init__(self, hidden_dim=512, encoder_name='gcn', gcn_layer=1,
                 num_head=4, gcn_attention=False, dropout=0.,
                 ):
        super(MultiGranularityVSEncoderV2, self).__init__()
        self.hid_dim = hidden_dim
        # self.co_attn = CoAttention(hidden_dim)

        self.node_embedding = nn.Embedding(2, hidden_dim)
        # nn.init.uniform_(self.node_embedding.weight, -1.732, 1.732)

        self.adj_leaner = AdjMatrixEdgeLearner(
            dim_input=hidden_dim,
            dim_hidden=hidden_dim
        )

        if encoder_name == 'gcn':
            self.encoder = GCNEncoder(hidden_dim, hidden_dim,
                                      num_hidden_layers=gcn_layer,
                                      dropout=dropout,
                                      attention=gcn_attention,
                                      embed_first=False)
        elif encoder_name == 'gat':
            self.encoder = GATEncoder(hidden_dim, hidden_dim // num_head,
                                      dropout=dropout, n_head=num_head)
        self.layer_norm = nn.LayerNorm(hidden_dim)

    def forward(self, vis_obj_embed, vis_img_embed
                ):
        """
        @param vis_obj_embed: [bs, n_img, D]
        @param vis_img_embed: [bs, n_img, D]
        @param sem_role_embed: [bs, n_know, D]
        @param sem_sent_embed: [bs, n_know, D]
        @return:
        """
        n_img = vis_obj_embed.shape[1]
        joint_embed = torch.cat([vis_obj_embed, vis_img_embed], dim=1)

        # [bs, 2*n_img + 2*n_know]
        node_type = torch.zeros(joint_embed.shape[:-1]).long().cuda(
            non_blocking=True)
        node_type[:, n_img:2 * n_img] = 1

        node_embed = self.node_embedding(node_type)
        # joint_embed = joint_embed + joint_embed * node_embed
        joint_embed = self.layer_norm(joint_embed * node_embed)

        joint_embed = self.encoder(joint_embed, self.adj_leaner(joint_embed))
        # return joint_embed.max(1)[0]
        return joint_embed.mean(1)
        # return joint_embed
