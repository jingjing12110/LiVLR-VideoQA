# @File: bilinear_attention.py
# @Github: https://github.com/jnhwkim/ban-vqa
# @Paper: Bilinear Attention Networks (https://arxiv.org/abs/1805.07932)

import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.weight_norm import weight_norm

from module.fc import BCNet


class BiAttention(nn.Module):
    def __init__(self, x_dim, y_dim, z_dim, glimpse=1, dropout=(.2, .5)):
        super(BiAttention, self).__init__()

        self.glimpse = glimpse
        self.logits = weight_norm(BCNet(x_dim, y_dim, z_dim, glimpse,
                                        dropout=dropout, k=3),
                                  name='h_mat', dim=None)

    def forward(self, v, q):
        """
        v: [bs, n_v, v_dim]
        q: [bs, n_q, q_dim]
        """
        v_num = v.size(1)
        q_num = q.size(1)

        logits = self.logits(v, q)  # [bs, ]

        p = F.softmax(logits.view(-1, self.glimpse, v_num * q_num), 2)
        p = p.view(-1, self.glimpse, v_num, q_num)

        return p


# class BiAttention(nn.Module):
#     def __init__(self, x_dim, y_dim, z_dim, glimpse=1, dropout=(.2, .5)):
#         super(BiAttention, self).__init__()
#
#         self.glimpse = glimpse
#         self.logits = weight_norm(BCNet(x_dim, y_dim, z_dim, glimpse,
#                                         dropout=dropout, k=3),
#                                   name='h_mat', dim=None)
#
#     def forward(self, v, q, v_mask=False):
#         """
#         v: [batch, k, vdim]
#         q: [batch, qdim]
#         """
#         p, logits = self.forward_all(v, q, v_mask)
#         return p, logits
#
#     def forward_all(self, v, q, v_mask=True, logit=False,
#                     mask_with=-float('inf')):
#         v_num = v.size(1)
#         q_num = q.size(1)
#
#         logits = self.logits(v, q)  # b x g x v x q
#         if v_mask:
#             mask = (0 == v.abs().sum(2)).unsqueeze(1).unsqueeze(3).expand(
#                 logits.size())
#             logits.data.masked_fill_(mask.data, mask_with)
#
#         p = F.softmax(logits.view(-1, self.glimpse, v_num * q_num), 2)
#         p = p.view(-1, self.glimpse, v_num, q_num)
#         if not logit:
#             return p, logits
#         return logits
