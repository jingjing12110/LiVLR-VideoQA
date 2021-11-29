# @File: graph_attention.py
# @Github: https://github.com/linjieli222/VQA_ReGAT
# @Paper: Relation-aware Graph Attention Network for Visual Question Answering

import math
import torch
import torch.nn as nn
from module.fc import FCNet
from torch.nn.utils.weight_norm import weight_norm


class GraphSelfAttentionLayer(nn.Module):
    def __init__(self, feat_dim, nongt_dim=20, pos_emb_dim=-1,
                 num_heads=16, dropout=(0.2, 0.5)):
        """ Attention module with vectorized version
        Args:
            position_embedding: [num_rois, nongt_dim, pos_emb_dim]
                                used in implicit relation
            pos_emb_dim: set as -1 if explicit relation
            nongt_dim: number of objects consider relations per image
            fc_dim: should be same as num_heads
            feat_dim: dimension of roi_feat
            num_heads: number of attention heads
        Returns:
            output: [num_rois, ovr_feat_dim, output_dim]
        """
        super(GraphSelfAttentionLayer, self).__init__()
        # multi head
        self.fc_dim = num_heads
        self.feat_dim = feat_dim
        self.dim = (feat_dim, feat_dim, feat_dim)
        self.dim_group = (int(self.dim[0] / num_heads),
                          int(self.dim[1] / num_heads),
                          int(self.dim[2] / num_heads))
        self.num_heads = num_heads
        self.pos_emb_dim = pos_emb_dim
        if self.pos_emb_dim > 0:
            self.pair_pos_fc1 = FCNet([pos_emb_dim, self.fc_dim], None,
                                      dropout[0])
        self.query = FCNet([feat_dim, self.dim[0]], None, dropout[0])
        self.nongt_dim = nongt_dim

        self.key = FCNet([feat_dim, self.dim[1]], None, dropout[0])

        self.linear_out_ = weight_norm(
            nn.Conv2d(in_channels=self.fc_dim * feat_dim,
                      out_channels=self.dim[2],
                      kernel_size=(1, 1),
                      groups=self.fc_dim), dim=None)

    def forward(self, roi_feat, adj_matrix,
                position_embedding, label_biases_att):
        """
        Args:
            :param roi_feat: [batch_size, N, feat_dim]
            :param adj_matrix: [batch_size, N, nongt_dim]
            :param position_embedding: [num_rois, nongt_dim, pos_emb_dim]
            :param label_biases_att:
        Returns:
            output: [batch_size, num_rois, ovr_feat_dim, output_dim]
        """
        batch_size = roi_feat.size(0)
        num_rois = roi_feat.size(1)
        nongt_dim = self.nongt_dim if self.nongt_dim < num_rois else num_rois
        # [batch_size, nongt_dim, feat_dim]
        nongt_roi_feat = roi_feat[:, :nongt_dim, :]

        # [batch_size,num_rois, self.dim[0] = feat_dim]
        q_data = self.query(roi_feat)

        # [batch_size, num_rois, num_heads, feat_dim /num_heads]
        q_data_batch = q_data.view(batch_size, num_rois, self.num_heads,
                                   self.dim_group[0])

        # [batch_size, num_heads, num_rois, feat_dim /num_heads]
        q_data_batch = torch.transpose(q_data_batch, 1, 2)

        # [batch_size, nongt_dim, self.dim[1] = feat_dim]
        k_data = self.key(nongt_roi_feat)

        # [batch_size,nongt_dim, num_heads, feat_dim /num_heads]
        k_data_batch = k_data.view(batch_size, nongt_dim, self.num_heads,
                                   self.dim_group[1])

        # [batch_size,num_heads, nongt_dim, feat_dim /num_heads]
        k_data_batch = torch.transpose(k_data_batch, 1, 2)

        # [batch_size, nongt_dim, feat_dim]
        v_data = nongt_roi_feat

        # [batch_size, num_heads, num_rois, nongt_dim]
        aff = torch.matmul(q_data_batch, torch.transpose(k_data_batch, 2, 3))

        # aff_scale, [batch_size, num_heads, num_rois, nongt_dim]
        aff_scale = (1.0 / math.sqrt(float(self.dim_group[1]))) * aff
        # aff_scale, [batch_size,num_rois,num_heads, nongt_dim]
        aff_scale = torch.transpose(aff_scale, 1, 2)
        weighted_aff = aff_scale

        if position_embedding is not None and self.pos_emb_dim > 0:
            # Adding goemetric features
            position_embedding = position_embedding.float()
            # [batch_size,num_rois * nongt_dim, emb_dim]
            position_embedding_reshape = position_embedding.view(
                (batch_size, -1, self.pos_emb_dim))

            # position_feat_1, [batch_size,num_rois * nongt_dim, fc_dim]
            position_feat_1 = self.pair_pos_fc1(position_embedding_reshape)
            position_feat_1_relu = nn.functional.relu(position_feat_1)

            # aff_weight, [batch_size,num_rois, nongt_dim, fc_dim]
            aff_weight = position_feat_1_relu.view(
                (batch_size, -1, nongt_dim, self.fc_dim))

            # aff_weight, [batch_size,num_rois, fc_dim, nongt_dim]
            aff_weight = torch.transpose(aff_weight, 2, 3)

            thresh = torch.FloatTensor([1e-6]).cuda()
            # weighted_aff, [batch_size,num_rois, fc_dim, nongt_dim]
            threshold_aff = torch.max(aff_weight, thresh)

            weighted_aff += torch.log(threshold_aff)

        if adj_matrix is not None:
            # weighted_aff_transposed:
            # [batch_size, num_rois, nongt_dim, num_heads]
            weighted_aff_transposed = torch.transpose(weighted_aff, 2, 3)
            zero_vec = -9e15 * torch.ones_like(weighted_aff_transposed)

            adj_matrix = adj_matrix.view(
                adj_matrix.shape[0], adj_matrix.shape[1],
                adj_matrix.shape[2], 1)
            adj_matrix_expand = adj_matrix.expand(
                (-1, -1, -1,
                 weighted_aff_transposed.shape[-1]))
            weighted_aff_masked = torch.where(adj_matrix_expand > 0,
                                              weighted_aff_transposed,
                                              zero_vec)

            weighted_aff_masked = weighted_aff_masked + \
                                  label_biases_att.unsqueeze(3)
            weighted_aff = torch.transpose(weighted_aff_masked, 2, 3)

        # aff_softmax, [batch_size, num_rois, fc_dim, nongt_dim]
        aff_softmax = nn.functional.softmax(weighted_aff, 3)

        # aff_softmax_reshape, [batch_size, num_rois*fc_dim, nongt_dim]
        aff_softmax_reshape = aff_softmax.view((batch_size, -1, nongt_dim))

        # output_t, [batch_size, num_rois * fc_dim, feat_dim]
        output_t = torch.matmul(aff_softmax_reshape, v_data)

        # output_t, [batch_size*num_rois, fc_dim * feat_dim, 1, 1]
        output_t = output_t.view((-1, self.fc_dim * self.feat_dim, 1, 1))

        # linear_out, [batch_size*num_rois, dim[2], 1, 1]
        linear_out = self.linear_out_(output_t)
        output = linear_out.view((batch_size, num_rois, self.dim[2]))

        return output


class GAttNet(nn.Module):
    def __init__(self, dir_num, label_num, in_feat_dim, out_feat_dim,
                 nongt_dim=20, dropout=0.2, label_bias=True, 
                 num_heads=16, pos_emb_dim=-1):
        """ Attention module with vectorized version
        Args:
            dir_num: number of edge directions
            label_num: number of edge labels
            in_feat_dim: dimension of roi_feat
            pos_emb_dim: dimension of position embedding for implicit relation,
            set as -1 for explicit relation
        Returns:
            output: [num_rois, ovr_feat_dim, output_dim]
        """
        super(GAttNet, self).__init__()
        assert dir_num <= 2, "Got more than two directions in a graph."
        self.dir_num = dir_num
        self.label_num = label_num
        self.in_feat_dim = in_feat_dim
        self.out_feat_dim = out_feat_dim
        self.nongt_dim = nongt_dim
        self.pos_emb_dim = pos_emb_dim

        self.dropout = nn.Dropout(dropout)
        self.self_weights = FCNet([in_feat_dim, out_feat_dim], '', dropout)
        self.bias = FCNet([label_num, 1], '', 0, label_bias)
        
        neighbor_net = []
        for i in range(dir_num):
            g_att_layer = GraphSelfAttentionLayer(
                                pos_emb_dim=pos_emb_dim,
                                num_heads=num_heads,
                                feat_dim=out_feat_dim,
                                nongt_dim=nongt_dim)
            neighbor_net.append(g_att_layer)
        self.neighbor_net = nn.ModuleList(neighbor_net)

    def forward(self, v_feat, adj_matrix, pos_emb=None):
        """
        Args:
            v_feat: [batch_size,num_rois, feat_dim]
            adj_matrix: [batch_size, num_rois, num_rois, num_labels]
            pos_emb: [batch_size, num_rois, pos_emb_dim]

        Returns:
            output: [batch_size, num_rois, feat_dim]
        """
        if self.pos_emb_dim > 0 and pos_emb is None:
            raise ValueError(
                f"position embedding is set to None "
                f"with pos_emb_dim {self.pos_emb_dim}")
        elif self.pos_emb_dim < 0 and pos_emb is not None:
            raise ValueError(
                f"position embedding is NOT None "
                f"with pos_emb_dim < 0")
        batch_size, num_rois, feat_dim = v_feat.shape
        nongt_dim = self.nongt_dim

        adj_matrix = adj_matrix.float()

        adj_matrix_list = [adj_matrix, adj_matrix.transpose(1, 2)]

        # Self - looping edges
        # [batch_size,num_rois, out_feat_dim]
        self_feat = self.self_weights(v_feat)

        output = self_feat
        neighbor_emb = [0] * self.dir_num
        for d in range(self.dir_num):
            # [batch_size,num_rois, nongt_dim,label_num]
            input_adj_matrix = adj_matrix_list[d][:, :, :nongt_dim, :]
            condensed_adj_matrix = torch.sum(input_adj_matrix, dim=-1)

            # [batch_size,num_rois, nongt_dim]
            v_biases_neighbors = self.bias(input_adj_matrix).squeeze(-1)

            # [batch_size,num_rois, out_feat_dim]
            neighbor_emb[d] = self.neighbor_net[d].forward(
                        self_feat, condensed_adj_matrix, pos_emb,
                        v_biases_neighbors)

            # [batch_size,num_rois, out_feat_dim]
            output = output + neighbor_emb[d]
        output = self.dropout(output)
        output = nn.functional.relu(output)

        return output
