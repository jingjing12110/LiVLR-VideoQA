# @File : adj_learner.py 
# @Time : 2020/8/20 
# @Email : jingjingjiang2017@gmail.com

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.weight_norm import weight_norm


class AdjMatrixLearner(nn.Module):
    def __init__(self, dim_input, dim_hidden, with_init=True):
        super(AdjMatrixLearner, self).__init__()
        self.with_init = with_init
        self.dim_input = dim_input

        self.edge_layer1 = weight_norm(nn.Linear(dim_input, dim_hidden), dim=None)
        self.edge_layer2 = weight_norm(nn.Linear(dim_input, dim_hidden), dim=None)
        self.attn_layer = weight_norm(nn.Linear(dim_input, 1), dim=None)

        self.reset_parameters()

    def reset_parameters(self):
        gain = nn.init.calculate_gain('relu')
        nn.init.xavier_normal_(self.edge_layer1.weight, gain=gain)
        nn.init.xavier_normal_(self.edge_layer2.weight, gain=gain)
        nn.init.xavier_normal_(self.attn_layer.weight, gain=gain)

    def forward(self, node_feat, adj_matrix):
        """
        @param node_feat: [bs, n_node, dim_feat]
        @param adj_matrix: [bs, n_node, n_node]
        @return:
        """
        n_node = node_feat.shape[1]
        f_self1 = F.relu(self.edge_layer1(node_feat))
        f_self2 = F.relu(self.edge_layer2(node_feat))
        adj_matrix_learn = F.normalize(
            torch.bmm(f_self1, f_self2.transpose(1, 2)), p=2, dim=1)

        if self.with_init:
            attention = node_feat.repeat(1, n_node, 1).view(-1, n_node, n_node,
                                                            self.dim_input)
            attention = self.attn_layer(attention).squeeze(dim=-1)
            attention = F.softmax(attention, dim=-1)  # [x, 10, 10]
            return adj_matrix_learn + torch.bmm(attention, adj_matrix)
        else:
            return adj_matrix_learn


class AdjMatrixEdgeLearner(nn.Module):
    def __init__(self, dim_input, dim_hidden, k=5, to_sparse=False):
        super(AdjMatrixEdgeLearner, self).__init__()
        self.dim_input = dim_input
        self.k = k
        self.to_sparse = to_sparse

        self.edge_layer1 = weight_norm(nn.Linear(dim_input, dim_hidden), dim=None)
        self.edge_layer2 = weight_norm(nn.Linear(dim_input, dim_hidden), dim=None)
        # self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_normal_(self.edge_layer1.weight)
        nn.init.kaiming_normal_(self.edge_layer2.weight)
        # gain = nn.init.calculate_gain('relu')
        # nn.init.xavier_normal_(self.edge_layer1.weight, gain=gain)
        # nn.init.xavier_normal_(self.edge_layer2.weight, gain=gain)

    def forward(self, node_feat):
        """
        @param node_feat: [bs, n_node, dim_feat]
        @return: [bs, n_node, n_node]
        """
        y1 = F.relu(self.edge_layer1(node_feat), inplace=True)
        y2 = F.relu(self.edge_layer1(node_feat), inplace=True)
        adj_matrix = torch.bmm(y1, y2.transpose(1, 2))

        if self.to_sparse:
            adj_matrix = F.softmax(F.normalize(adj_matrix, dim=-1), dim=-1)
            # retain the graph sparsity
            _, top_ind = torch.topk(adj_matrix, k=self.k, dim=-1,
                                    largest=False, sorted=False)
            adj_matrix = adj_matrix.scatter(2, top_ind, 0.)
            adj_matrix = F.normalize(adj_matrix, dim=-1, p=1)

        return adj_matrix


# class AdjMatrixEdgeLearner(nn.Module):
#     def __init__(self, dim_input, dim_hidden, with_init=True):
#         super(AdjMatrixEdgeLearner, self).__init__()
#         self.with_init = with_init
#         self.dim_input = dim_input
#
#         self.edge_conv = nn.Sequential(weight_norm(nn.Conv1d(64, 1, 1), dim=None),
#                                        nn.ReLU(inplace=True))
#         self.edge_layer = weight_norm(nn.Linear(dim_input, dim_hidden), dim=None)
#         self.reset_parameters()
#
#     def reset_parameters(self):
#         gain = nn.init.calculate_gain('relu')
#         nn.init.xavier_normal_(self.edge_conv.weight, gain=gain)
#         nn.init.xavier_normal_(self.edge_layer.weight, gain=gain)
#
#     def forward(self, node_feat, po_embed, adj_matrix):
#         """
#         @param node_feat: [bs, n_node, dim_feat]
#         @param adj_matrix: [bs, n_node, n_node]
#         @param po_embed: [bs, n_obj, n_obj, 64]
#         @return:
#         """
#         bs, n_node, _ = node_feat.shape
#         adj_learn = self.edge_conv(
#             po_embed.permute(0, 2, 1)).squeeze().view(-1, n_node, n_node)
#
#         y = F.relu(self.edge_layer(node_feat))
#         adj_sim = F.softmax(
#             F.normalize(torch.bmm(y, y.transpose(1, 2)), p=2, dim=1), dim=-1)
#
#         if self.with_init:
#             adj_matrix = adj_learn + adj_sim.masked_fill(adj_matrix == 0, -9e15)
#             return adj_matrix
#         else:
#             return torch.bmm(adj_learn, adj_sim)


# class AdjMatrixEdgeLearner(nn.Module):
#     def __init__(self, dim_input, dim_hidden, with_init=True):
#         super(AdjMatrixEdgeLearner, self).__init__()
#         self.with_init = with_init
#         self.dim_input = dim_input
#         self.k = 4
#
#         self.edge_conv = nn.Sequential(
#             weight_norm(nn.Conv1d(dim_input * 2, dim_hidden, 1)),
#             nn.BatchNorm1d(dim_hidden), nn.ReLU(inplace=True))
#
#         self.edge_layer = weight_norm(nn.Linear(dim_input, dim_hidden))
#
#         # self.attn_layer = weight_norm(nn.Linear(dim_input, 1))
#
#     def forward(self, node_feat, adj_matrix):
#         """
#         @param node_feat: [bs, n_node, dim_feat]
#         @param adj_matrix: [bs, n_node, n_node]
#         @return:
#         """
#         bs, n_node, dim_feat = node_feat.shape
#
#         KNN_Graph = self.create_knn_graph(node_feat)
#         x = node_feat.view([bs, n_node, 1, dim_feat]).repeat(1, 1, self.k, 1)
#
#         x = torch.cat([x, KNN_Graph - x], dim=3).permute(0, 3, 1, 2)
#         # [bs, 2*dim_feat, n_node, k] -> [bs, 2*dim_feat, n_node * k]
#         x = x.view([bs, 2 * dim_feat, n_node * self.k])
#         x = self.edge_conv(x)  # [bs, -1, n_node * k]
#         x = nn.MaxPool1d(self.k)(
#             x.view(bs, -1, n_node, self.k).view(bs, -1, self.k))
#         x = x.view(bs, -1, n_node).permute(0, 2, 1)
#
#         y = F.relu(self.edge_layer(node_feat))
#
#         adj_matrix_learn = F.normalize(
#             torch.bmm(x, y.transpose(1, 2)), p=2, dim=1)
#
#         if self.with_init:
#             # f_neighbor = node_feat.repeat(1, n_node, 1).view(-1, n_node, n_node,
#             #                                                  self.input_dim)
#             # attention = self.attn_layer(f_neighbor).squeeze(dim=-1)
#             # attention = F.softmax(attention, dim=-1)  # [x, 10, 10]
#             # adj_matrix_learn = adj_matrix_learn + attention * adj_matrix
#             return adj_matrix_learn + adj_matrix
#         else:
#             return adj_matrix_learn
#
#     def create_knn_graph(self, x):
#         # 计算距离矩阵
#         # n_node = x.shape[0]
#         # dist_mat = torch.pow(x, 2).sum(dim=1, keepdim=True).repeat(1, n_node)
#         # dist_mat = dist_mat + dist_mat.t()
#         # dist_mat = dist_mat.addmm(beta=1, alpha=-2, mat1=x, mat2=x.t())
#         x_t = torch.stack([x[i].t() for i in range(x.shape[0])], dim=0)
#         dist_mat = torch.bmm(x, x_t)
#
#         # 对距离矩阵排序
#         _, sorted_indices = torch.sort(dist_mat, dim=2)
#
#         # 取出前K个（除去本身）
#         knn_indexes = sorted_indices[:, :, 1:self.k + 1]
#         # 创建KNN图
#         knn_graph = torch.stack([x[i][knn_indexes[i]] for i in range(x.shape[0])],
#                                 dim=0)
#
#         return knn_graph
