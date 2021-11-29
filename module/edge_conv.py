# @File : edge_conv.py 
# @Time : 2020/8/19 
# @Email : jingjingjiang2017@gmail.com

from collections import OrderedDict

import torch
import torch.nn as nn


def conv_bn_block(input, output, kernel_size):
    """ 标准卷积块（conv + bn + relu）
    :param input: 输入通道数
    :param output: 输出通道数
    :param kernel_size: 卷积核大小
    :return:
    """
    return nn.Sequential(
        nn.Conv1d(input, output, kernel_size),
        nn.BatchNorm1d(output),
        nn.ReLU(inplace=True)
    )


class EdgeConv(nn.Module):
    """ EdgeConv模块
    1. 输入为：n * f
    2. 创建KNN graph，变为： n * k * f
    3. 接上若干个mlp层：a1, a2, ..., an
    4. 最终输出为：n * k * an
    5. 全局池化，变为： n * an
    """

    def __init__(self, layers, K=20):
        """构造函数
        :param layers: e.p. [3, 64, 64, 64]
        :param K:"""
        super(EdgeConv, self).__init__()

        self.K = K
        self.layers = layers

        if layers is None:
            self.mlp = None
        else:
            mlp_layers = OrderedDict()
            for i in range(len(self.layers) - 1):
                if i == 0:
                    mlp_layers['conv_bn_block_{}'.format(i + 1)] = conv_bn_block(
                        2 * self.layers[i], self.layers[i + 1], 1)
                else:
                    mlp_layers['conv_bn_block_{}'.format(i + 1)] = conv_bn_block(
                        self.layers[i], self.layers[i + 1], 1)
            self.mlp = nn.Sequential(mlp_layers)

    def createSingleKNNGraph(self, X):
        """ generate a KNN graph for a single point cloud
        :param X:  X is a Tensor, shape: [N, F]
        :return: KNN graph, shape: [N, K, F]
        """
        N, F = X.shape
        assert F == self.layers[0]

        # 计算距离矩阵
        dist_mat = torch.pow(X, 2).sum(dim=1, keepdim=True).expand(
            N, N) + torch.pow(X, 2).sum(dim=1, keepdim=True).expand(N, N).t()
        dist_mat.addmm_(1, -2, X, X.t())

        # 对距离矩阵排序
        dist_mat_sorted, sorted_indices = torch.sort(dist_mat, dim=1)
        # print(dist_mat_sorted)

        # 取出前K个（除去本身）
        knn_indexes = sorted_indices[:, 1:self.K + 1]
        # print(sorted_indices)

        # 创建KNN图
        knn_graph = X[knn_indexes]

        return knn_graph

    def forward(self, X):
        """
        前向传播函数
        :param X:  shape: [B, N, F]
        :return:  shape: [B, N, an]
        """
        B, N, F = X.shape
        assert F == self.layers[0]

        KNN_Graph = torch.zeros(B, N, self.K, self.layers[0]).to(X.device)

        # creating knn graph
        # X: [B, N, F]
        for idx, x in enumerate(X):
            # x: [N, F]
            # knn_graph: [N, K, F]
            # self.KNN_Graph[idx] = self.createSingleKNNGraph(x)
            KNN_Graph[idx] = self.createSingleKNNGraph(x)
        # print(self.KNN_Graph.shape)
        # print('KNN_Graph: {}'.format(KNN_Graph[0][0]))

        # X: [B, N, F]
        x1 = X.reshape([B, N, 1, F])
        x1 = x1.expand(B, N, self.K, F)
        # x1: [B, N, K, F]

        x2 = KNN_Graph - x1
        # x2: [B, N, K, F]

        x_in = torch.cat([x1, x2], dim=3)
        # x_in: [B, N, K, 2*F]
        x_in = x_in.permute(0, 3, 1, 2)
        # x_in: [B, 2*F, N, K]

        # reshape, x_in: [B, 2*F, N*K]
        x_in = x_in.reshape([B, 2 * F, N * self.K])

        # out: [B, an, N*K]
        out = self.mlp(x_in)
        _, an, _ = out.shape
        # print(out.shape)

        out = out.reshape([B, an, N, self.K])
        # print(out.shape)
        # reshape, out: [B, an, N, K]
        out = out.reshape([B, an * N, self.K])
        # print(out.shape)
        # reshape, out: [B, an*N, K]
        out = nn.MaxPool1d(self.K)(out)
        # print(out.shape)
        out = out.reshape([B, an, N])
        # print(out.shape)
        out = out.permute(0, 2, 1)
        # print(out.shape)

        return out
