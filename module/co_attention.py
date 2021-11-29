import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.weight_norm import weight_norm


class CoAttention(nn.Module):
    """
    Co-Attention mechanism for visual and semantic features.
    """
    def __init__(self, hidden_dim, sum=False):
        super(CoAttention, self).__init__()
        self.hidden_dim = hidden_dim
        self.sum = sum

        # Affinity layer
        self.W_b = weight_norm(nn.Linear(self.hidden_dim, self.hidden_dim))

        # Attention layers
        self.W_v = weight_norm(nn.Linear(self.hidden_dim, self.hidden_dim))
        self.W_q = weight_norm(nn.Linear(self.hidden_dim, self.hidden_dim))

        self.w_v = weight_norm(nn.Linear(self.hidden_dim, 1))
        self.w_q = weight_norm(nn.Linear(self.hidden_dim, 1))

    def forward(self, v, q):
        """
        :param v: [bs, N, 512]
        :param q: [bs, L, 512]
        :returns: [bs, 512], [bs, 512]
        """
        # Affinity matrix
        C = torch.tanh(torch.bmm(q, v.permute(0, 2, 1)))  # [bs, L, N]
        a_v = torch.tanh(self.W_v(v) + torch.bmm(C.transpose(2, 1), self.W_q(q)))
        a_q = torch.tanh(self.W_q(q) + torch.bmm(C, self.W_v(v)))
        # C = torch.bmm(q, v.permute(0, 2, 1))  # [bs, L, N]
        # H_v = self.W_v(v) + torch.bmm(C.transpose(2, 1), self.W_q(q))
        # H_q = self.W_q(q) + torch.bmm(C, self.W_v(v))

        # Attention weights
        a_v = F.softmax(self.w_v(a_v), dim=1)  # [bs, N, 1]
        a_q = F.softmax(self.w_q(a_q), dim=1)  # [bs, L, 1]

        # Compute attention-weighted features
        if self.sum:
            v = torch.mean(a_v * v, dim=1)  # [bs, hidden_dim]
            q = torch.mean(a_q * q, dim=1)  # [bs, hidden_dim]
        else:
            v = a_v * v  # [bs, N, hidden_dim]
            q = a_q * q  # [bs, L, hidden_dim]

        return v, q
