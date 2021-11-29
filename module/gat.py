# @File : gat.py
# @Github : https://github.com/Diego999/pyGAT

import torch
import torch.nn as nn
import torch.nn.functional as F


class GraphAttention(nn.Module):
    """
    Simple GAT layer, similar to https://arxiv.org/abs/1710.10903
    """

    def __init__(self, in_features, out_features, dropout, alpha, concat=True):
        super(GraphAttention, self).__init__()
        self.dropout = dropout
        self.concat = concat

        self.W = nn.Parameter(torch.zeros(size=(in_features, out_features)),
                              requires_grad=True)
        self.a1 = nn.Parameter(torch.zeros(size=(in_features, 1)),
                               requires_grad=True)
        self.a2 = nn.Parameter(torch.zeros(size=(in_features, 1)),
                               requires_grad=True)
        self.reset_parameters()

        self.leakyrelu = nn.LeakyReLU(alpha)

    def reset_parameters(self):
        nn.init.xavier_normal_(self.W.data, gain=1.414)
        nn.init.xavier_normal_(self.a1.data, gain=1.414)
        nn.init.xavier_normal_(self.a2.data, gain=1.414)

    def forward(self, x, adj):
        """
        @param x: [n_node, dim_feat]
        @param adj: [n_node, n_node]
        @return:
        """
        h = torch.mm(x, self.W)

        f_1 = torch.matmul(h, self.a1)
        f_2 = torch.matmul(h, self.a2)
        e = self.leakyrelu(f_1 + f_2.transpose(0, 1))

        zero_vec = -9e15 * torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=1)
        attention = F.dropout(attention, self.dropout, training=self.training)
        h_prime = torch.matmul(attention, h)

        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime


# class GraphAttentionLayer(nn.Module):
#     """
#     GAT layer with batch, similar to https://arxiv.org/abs/1710.10903
#     """
#
#     def __init__(self, in_features, out_features, dropout, alpha=0.2,
#                  concat=True):
#         super(GraphAttentionLayer, self).__init__()
#         self.dropout = dropout
#         self.concat = concat
#
#         self.W = nn.Parameter(torch.zeros(size=(in_features, out_features)),
#                               requires_grad=True)
#         nn.init.xavier_uniform_(self.W.data, gain=1.414)
#         self.a = nn.Parameter(torch.zeros(size=(2 * out_features, 1)),
#                               requires_grad=True)
#         nn.init.xavier_uniform_(self.a.data, gain=1.414)
#
#         self.leakyrelu = nn.LeakyReLU(alpha)
#
#     def forward(self, x, adj):
#         """
#         @param x: [bs, num_node, feat_dim]
#         @param adj: [bs, num_node, num_node]
#         @return: [bs, num_node, feat_dim]
#         """
#         # h = torch.mm(input_x, self.W)
#         bs, N, D = x.shape
#         h = torch.matmul(x, self.W)  # [bs, N, D]
#         # 这里让每两个节点的向量都连接在一起遍历一次得到 N * N * (2 * out_features)大小的矩阵
#         a_input = torch.cat(  # [bs, N, N, 2D]
#             [h.repeat(1, 1, N).view(bs, N * N, -1), h.repeat(1, N, 1)],
#             dim=2).view(bs, N, -1, 2 * self.out_features)
#         # [bs, N, N]
#         attention = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(3))
#
#         # Masked Attention
#         zero_vec = -9e15 * torch.ones_like(attention)
#         attention = torch.where(adj > 0, attention, zero_vec)
#
#         # 这里是一个非线性变换，将有权重的变得更趋近于1，没权重的为0
#         attention = F.softmax(attention, dim=2)  # [bs, N, N]
#         attention = F.dropout(attention, self.dropout, training=self.training)
#         # h_prime = torch.matmul(attention, h)
#         h_prime = torch.bmm(attention, h)  # [B,N,N]*[B,N,C] -> [B,N,C]
#
#         if self.concat:
#             return F.elu(h_prime)
#         else:
#             return h_prime


class GraphAttentionLayer(nn.Module):
    """
    GAT layer with batch, similar to https://arxiv.org/abs/1710.10903
    """

    def __init__(self, dim_input, dim_hidden, dropout, alpha=0.2,
                 concat=True):
        super(GraphAttentionLayer, self).__init__()
        self.dropout = dropout
        self.concat = concat
        self.dim_hidden = dim_hidden

        self.linear_layer = nn.Linear(dim_input, dim_hidden, bias=False)
        self.attn_layer = nn.Linear(2 * dim_hidden, 1, bias=False)
        self.reset_parameters()

        self.leaky_relu = nn.LeakyReLU(alpha)

    def reset_parameters(self):
        gain = nn.init.calculate_gain('relu')
        nn.init.xavier_normal_(self.linear_layer.weight, gain=gain)
        nn.init.xavier_normal_(self.attn_layer.weight, gain=gain)

    def forward(self, x, adj):
        """
        @param x: [bs, num_node, feat_dim]
        @param adj: [bs, num_node, num_node]
        @return: [bs, num_node, feat_dim]
        """
        bs, N, _ = x.shape
        h = self.linear_layer(x)  # [bs, N, dim_hidden]

        h_self = h.repeat(1, 1, N).view(bs, N * N, -1)
        h_neighbor = h.repeat(1, N, 1)
        a_input = torch.cat([h_self, h_neighbor], dim=2).view(bs, N, -1,
                                                              2 * self.dim_hidden)
        # [bs, N, N]
        attention = self.leaky_relu(self.attn_layer(a_input)).squeeze(dim=-1)

        # Masked Attention
        attention = attention.masked_fill(adj == 0, -9e15)
        # zero_vec = -9e15 * torch.ones_like(attention)
        # attention = torch.where(adj > 0, attention, zero_vec)

        attention = F.softmax(attention, dim=-1)  # [bs, N, N]

        # [bs, N, C]
        if self.concat:
            return F.elu(torch.bmm(attention, h))
        else:
            return torch.bmm(attention, h)


class GATEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, n_head, dropout=0.,
                 alpha=0.2, merge='cat'):
        """
        @param input_dim: [num_node, in_dim]
        @param hidden_dim: [num_node, hidden_dim]
        @param dropout:
        @param alpha:
        @param n_head:
        """
        super(GATEncoder, self).__init__()
        self.dropout = dropout
        self.merge = merge

        self.gat_layers = nn.ModuleList()
        for _ in range(n_head):
            self.gat_layers.append(GraphAttentionLayer(
                input_dim, hidden_dim, dropout=dropout, alpha=alpha, concat=True))

    def forward(self, x, adj):
        x = F.dropout(x, self.dropout, training=self.training)
        if self.merge == 'cat':
            x = torch.cat([att(x, adj) for att in self.gat_layers], dim=2)
        else:
            x = torch.mean(torch.stack([att(x, adj) for att in self.gat_layers]))

        return x


class GAT(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, nheads, alpha=0.2):
        """ Dense version of GAT.
        @param nfeat: [num_node, in_dim]
        @param nhid: [num_node, hidden_dim]
        @param nclass:
        @param dropout:
        @param alpha:
        @param nheads:
        """
        super(GAT, self).__init__()
        self.dropout = dropout

        self.attentions = [
            GraphAttentionLayer(nfeat, nhid, dropout=dropout, alpha=alpha,
                                concat=True) for _ in range(nheads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

        self.out_att = GraphAttentionLayer(nhid * nheads, nclass, dropout=dropout,
                                           alpha=alpha, concat=False)

    def forward(self, x, adj):
        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.cat([att(x, adj) for att in self.attentions], dim=2)
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.elu(self.out_att(x, adj))
        return F.log_softmax(x, dim=1)
