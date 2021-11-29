# @File : gcn.py 
# @Time : 2020/8/17 
# @Email : jingjingjiang2017@gmail.com

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.weight_norm import weight_norm


class GraphConvolution(nn.Module):
    """original author: """

    def __init__(self, in_feature, out_feature, bias=True):
        super(GraphConvolution, self).__init__()
        self.weight = nn.Parameter(torch.FloatTensor(in_feature, out_feature),
                                   requires_grad=True)
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_feature),
                                     requires_grad=True)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, node_feat, adj):
        """
        @param node_feat: [n_node, dim_feat]
        @param adj: [n_node, n_node]
        @return:
        """
        support = torch.mm(node_feat, self.weight)
        output = torch.spmm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output


class GCNLayer(nn.Module):
    def __init__(self, dim_input, dim_hidden, dropout=0.0):
        super(GCNLayer, self).__init__()
        self.ctx_layer = nn.Linear(dim_input, dim_hidden, bias=False)
        self.layer_norm = nn.LayerNorm(dim_hidden)
        self.dropout = nn.Dropout(p=dropout, inplace=True)

    def forward(self, node_fts, rel_edges):
        """
        @param node_fts: (bs, num_nodes, embed_size)
        @param rel_edges: (bs, num_nodes, num_nodes)
        @return:
        """
        ctx_embeds = self.ctx_layer(torch.bmm(rel_edges, node_fts))
        node_embeds = node_fts + self.dropout(ctx_embeds)
        node_embeds = self.layer_norm(node_embeds)
        return node_embeds


class AttnGCNLayer(GCNLayer):
    def __init__(self, embed_size, d_ff, dropout=0.0):
        super(AttnGCNLayer, self).__init__(embed_size, embed_size, dropout)
        self.edge_attn_query = nn.Linear(embed_size, d_ff)
        self.edge_attn_key = nn.Linear(embed_size, d_ff)
        self.attn_denominator = math.sqrt(d_ff)

    def forward(self, node_fts, rel_edges):
        # (bs, num_nodes, num_nodes)
        attn_scores = torch.einsum('bod,bid->boi',
                                   self.edge_attn_query(node_fts),
                                   self.edge_attn_key(node_fts)
                                   ) / self.attn_denominator
        attn_scores = attn_scores.masked_fill(rel_edges == 0, -1e10)
        attn_scores = torch.softmax(attn_scores, dim=2)
        # some nodes do not connect with any edge
        attn_scores = attn_scores.masked_fill(rel_edges == 0, 0)

        ctx_embeds = self.ctx_layer(torch.bmm(attn_scores, node_fts))
        node_embeds = node_fts + self.dropout(ctx_embeds)
        node_embeds = self.layer_norm(node_embeds)
        return node_embeds


class GCNEncoder(nn.Module):
    def __init__(self, dim_input, dim_hidden, num_hidden_layers,
                 embed_first=False, attention=False, dropout=0.):
        super(GCNEncoder, self).__init__()
        self.num_hidden_layers = num_hidden_layers
        self.embed_first = embed_first

        if self.embed_first:
            self.first_embedding = nn.Sequential(
                nn.LayerNorm(dim_input),
                weight_norm(nn.Linear(dim_input, dim_hidden)),
                nn.LayerNorm(dim_hidden),
                nn.ReLU(inplace=True),
                nn.Dropout(p=dropout),
            )
            # self.reset_parameters()
            # nn.init.kaiming_normal_(self.first_embedding[1].weight)

        if attention:
            gcn_fn = AttnGCNLayer
        else:
            gcn_fn = GCNLayer

        self.layers = nn.ModuleList()
        for k in range(num_hidden_layers):
            if attention:
                h2h = gcn_fn(dim_hidden, dim_hidden // 2, dropout=dropout)
            else:
                h2h = gcn_fn(dim_hidden, dim_hidden, dropout=dropout)
            self.layers.append(h2h)

    def reset_parameters(self):
        gain = nn.init.calculate_gain('relu')
        nn.init.xavier_normal_(self.first_embedding[1].weight, gain=gain)

    def forward(self, node_fts, rel_edges):
        if self.embed_first:
            node_fts = self.first_embedding(node_fts)

        for k in range(self.num_hidden_layers):
            layer = self.layers[k]
            node_fts = layer(node_fts, rel_edges)
        # (bs, num_nodes, dim_hidden)
        return node_fts


class GPool(nn.Module):
    def __init__(self, k, in_dim, p=0.):
        super(GPool, self).__init__()
        self.k = k
        self.sigmoid = nn.Sigmoid()
        self.proj = nn.Linear(in_dim, 1)
        self.drop = nn.Dropout(p=p, inplace=True) if p > 0 else nn.Identity()

    def forward(self, node_fts, rel_edges):
        weight = self.proj(self.drop(node_fts)).squeeze()
        value, idx = torch.topk(self.sigmoid(weight), self.k)
        node_fts = torch.stack([node_fts[i][idx.squeeze()[i]] for i in
                                range(node_fts.shape[0])], dim=0)

        return torch.mul(node_fts, value)

    def top_k_node(self, scores, g, h):
        value, idx = torch.topk(scores, self.k)
        new_h = torch.stack(
            [h[i][idx.squeeze()[i]] for i in range(len(h.shape[0]))], dim=0)
        new_h = torch.mul(new_h, value)
        un_g = g.bool().float()
        un_g = torch.matmul(un_g, un_g).bool().float()
        un_g = un_g[idx, :]
        un_g = un_g[:, idx]
        g = un_g / torch.sum(un_g, 1)  # norm g
        return g, new_h, idx
