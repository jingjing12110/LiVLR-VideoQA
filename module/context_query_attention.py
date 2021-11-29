# @File : context_query_attention.py
# @Time : 2020/7/23 
# @Email : jingjingjiang2017@gmail.com

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class ContextQueryAttention(nn.Module):
    """TVQAPlus
        Use each word in context (qa) to attend to words in query (cap/video).
    """

    def __init__(self, dropout=0.1):
        super(ContextQueryAttention, self).__init__()
        self.dropout = dropout

    def similarity(self, query, query_mask, context, context_mask):
        context = F.dropout(F.normalize(context, p=2, dim=-1), p=self.dropout,
                            training=self.training)  # notice
        query = F.dropout(F.normalize(query, p=2, dim=-1), p=self.dropout,
                          training=self.training)

        # compute cosine score
        s_mask = torch.mul(context_mask.unsqueeze(-1), query_mask.unsqueeze(-2))
        s = torch.matmul(context, query.transpose(-2, -1)) / math.sqrt(
            context.shape[-1])
        s = s - 1e10 * (1. - s_mask)

        return s, s_mask

    def forward(self, query, query_mask, context, context_mask):
        """
        @param query: [bs, 1, n_cap, n_verb, 512] // cap
        @param query_mask:
        @param context: [bs, 1, n_cap, n_word, 512]  // qa_bert
        @param context_mask:
        @return:
        """
        # [bs, 1, n_cap, n_word, n_verb]
        s, s_mask = self.similarity(query, query_mask, context, context_mask)
        s_ = F.softmax(s, dim=-1)
        s_ = s_ * s_mask

        # qa-ware cap/visual representation [bs, 1, n_cap, n_word, 512]
        return torch.matmul(s_, query)


class ContextAttention(nn.Module):
    """ refine for TVQAPlus
    """

    def __init__(self, hidden_dim):
        super(ContextAttention, self).__init__()
        self.cq_att = ContextQueryAttention()

        self.c2q_down_projection = nn.Sequential(
            nn.LayerNorm(3 * hidden_dim),
            nn.Dropout(p=0.1),
            nn.Linear(3 * hidden_dim, hidden_dim),
            nn.ReLU(True)
        )

    def forward(self, query, query_mask, context, context_mask):
        context = context.unsqueeze(dim=1).repeat(1, query.shape[1], 1,
                                                  1).unsqueeze(dim=1)
        context_mask = context_mask.unsqueeze(dim=1).repeat(1, query.shape[1],
                                                            1).unsqueeze(dim=1)
        # context = context.unsqueeze(dim=1).unsqueeze(dim=1)
        # context_mask = context_mask.unsqueeze(dim=1).unsqueeze(dim=1)
        query = query.unsqueeze(dim=1)
        query_mask = query_mask.unsqueeze(dim=1)

        num_img, num_obj = query_mask.shape[2:]
        # [bs, 1, n_cap, n_word, 512]
        u_a = self.cq_att(query, query_mask, context, context_mask)
        mixed = torch.cat([context, u_a, context * u_a], dim=-1)
        mixed = self.c2q_down_projection(mixed)  # (bs, Lqa, D)

        return mixed
