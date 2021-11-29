# @File : general_attention.py
# @Time : 2020/1/16
# @Email : jingjingjiang2017@gmail.com

import numpy as np
import torch
import torch.nn as nn
from module.fc import FCNet
import torch.nn.functional as F


class CALayer(nn.Module):
    """Channel Attention (CA) Layer
    """
    def __init__(self, channel, reduction=16):
        super(CALayer, self).__init__()
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1,
                      padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1,
                      padding=0, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y


class SpatialAttn(nn.Module):
    """Spatial Attention Layer
        https://github.com/HRanWang/Spatial-Attention
    """
    def __init__(self):
        super(SpatialAttn, self).__init__()

    def forward(self, x):
        # global cross-channel averaging # [bs, channel, w, h]
        x = x.mean(1, keepdim=True)  # [bs, 1, w, h]
        h = x.size(2)
        w = x.size(3)
        x = x.view(x.size(0), -1)  # [bs, wh]
        z = x
        for b in range(x.size(0)):
            z[b] /= torch.sum(z[b])
        z = z.view(x.size(0), 1, h, w)  # [bs, 1, w, h]
        return z


class SelfAttn(nn.Module):
    """ CNN-based Self attention Layer
        https://github.com/heykeetae/Self-Attention-GAN
    """
    def __init__(self, in_dim, activation):
        super(SelfAttn, self).__init__()
        self.chanel_in = in_dim
        self.activation = activation

        self.query_conv = nn.Conv2d(in_channels=in_dim,
                                    out_channels=in_dim // 8,
                                    kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim,
                                  out_channels=in_dim // 8,
                                  kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim,
                                    out_channels=in_dim,
                                    kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1),
                                  requires_grad=True)

        self.softmax = nn.Softmax(dim=-1)  #

    def forward(self, x):
        """output self attention value + input feature
        @param x: input feature maps( B X C X W X H)
        @return: attention: B X N X N (N is Width*Height)
        """
        bs, C, width, height = x.size()
        proj_query = self.query_conv(x).view(bs, -1, width * height
                                             ).permute(0, 2, 1)  # B X CX(N)
        proj_key = self.key_conv(x).view(bs, -1,
                                         width * height)  # B X C x (*W*H)
        energy = torch.bmm(proj_query, proj_key)  # transpose check
        attention = self.softmax(energy)  # BX (N) X (N)
        proj_value = self.value_conv(x).view(bs, -1,
                                             width * height)  # B X C X N

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(bs, C, width, height)

        out = self.gamma * out + x

        return out, attention


class CoAttention(nn.Module):
    def __init__(self, v_dim, q_dim, hid_dim, dropout=(.2, .5)):
        super(CoAttention, self).__init__()
        self.hid_dim = hid_dim
        act = "ReLU"
        self.v_net = FCNet([v_dim, self.hid_dim], act=act, dropout=dropout[0])
        self.q_net = FCNet([q_dim, self.hid_dim], act=act, dropout=dropout[0])

    def forward(self, v, q, v_mask=True):
        """
        @param v: video feature [bs, frame_num, dim]
        @param q: question feature [bs, q_len, dim]
        @param v_mask:
        @return: attention feature (residual connection)
        """
        v_fc = self.v_net(v)
        q_fc = self.q_net(q)
        aff_mat = torch.matmul(v_fc, q_fc.transpose(1, 2))
        v_att = F.softmax(aff_mat, dim=1)[:, :, 0].unsqueeze(2)
        q_att = F.softmax(aff_mat, dim=2)[:, 0, :].unsqueeze(2)
        v_attend = (v * v_att) + v
        q_attend = (q * q_att) + q
        return v_attend, q_attend


class CoAttention2(nn.Module):
    def __init__(self, v_dim, q_dim, hid_dim, dropout=(.2, .5)):
        super(CoAttention2, self).__init__()
        self.hid_dim = hid_dim
        act = "ReLU"
        self.v_net = FCNet([v_dim, self.hid_dim], act=act, dropout=dropout[0])
        self.q_net = FCNet([q_dim, self.hid_dim], act=act, dropout=dropout[0])

    def forward(self, v, q, v_mask=True):
        """
        @param v: video feature [bs, frame_num, dim]
        @param q: question feature [bs, q_len, dim]
        @param v_mask:
        @return: attention feature (residual connection)
        """
        v_fc = self.v_net(v)
        q_fc = self.q_net(q)
        aff_mat = torch.matmul(v_fc, q_fc.transpose(1, 2))
        v_att = F.softmax(aff_mat, dim=1)[:, :, 0].unsqueeze(2)
        q_att = F.softmax(aff_mat, dim=2)[:, 0, :].unsqueeze(2)
        v_attend = (v * v_att) + v
        q_attend = (q * q_att) + q
        return v_attend, q_attend


class ScaledDotProductAttention(nn.Module):
    """Scaled Dot-Product Attention
        https://arxiv.org/abs/1706.03762
    """
    def __init__(self, temperature, attn_dropout=0.1):
        super(ScaledDotProductAttention, self).__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, q, k, v, mask=None):
        attn = torch.bmm(q, k.transpose(1, 2))
        attn = attn / self.temperature

        if mask is not None:
            attn = attn.masked_fill(mask, -np.inf)

        attn = self.softmax(attn)
        attn = self.dropout(attn)
        output = torch.bmm(attn, v)
        return output, attn


class MultiHeadAttention(nn.Module):
    """Multi-Head Attention module
        https://arxiv.org/abs/1706.03762
    """
    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.w_qs = nn.Linear(d_model, n_head * d_k)
        self.w_ks = nn.Linear(d_model, n_head * d_k)
        self.w_vs = nn.Linear(d_model, n_head * d_v)
        nn.init.normal_(self.w_qs.weight, mean=0,
                        std=np.sqrt(2.0 / (d_model + d_k)))
        nn.init.normal_(self.w_ks.weight, mean=0,
                        std=np.sqrt(2.0 / (d_model + d_k)))
        nn.init.normal_(self.w_vs.weight, mean=0,
                        std=np.sqrt(2.0 / (d_model + d_v)))

        self.attention = ScaledDotProductAttention(
            temperature=np.power(d_k, 0.5))
        self.layer_norm = nn.LayerNorm(d_model)

        self.fc = nn.Linear(n_head * d_v, d_model)
        nn.init.xavier_normal_(self.fc.weight)

        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, mask=None):
        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head

        sz_b, len_q, _ = q.size()
        sz_b, len_k, _ = k.size()
        sz_b, len_v, _ = v.size()

        residual = q

        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)

        q = q.permute(2, 0, 1, 3).contiguous().view(-1, len_q,
                                                    d_k)  # (n*b) x lq x dk
        k = k.permute(2, 0, 1, 3).contiguous().view(-1, len_k,
                                                    d_k)  # (n*b) x lk x dk
        v = v.permute(2, 0, 1, 3).contiguous().view(-1, len_v,
                                                    d_v)  # (n*b) x lv x dv

        mask = mask.repeat(n_head, 1, 1)  # (n*b) x .. x ..
        output, attn = self.attention(q, k, v, mask=None)

        output = output.view(n_head, sz_b, len_q, d_v)
        # b x lq x (n*dv)
        output = output.permute(1, 2, 0, 3).contiguous().view(sz_b, len_q, -1)

        output = self.dropout(self.fc(output))
        output = self.layer_norm(output + residual)
        return output, attn


class GuidedAttention(nn.Module):
    def __init__(self, F_hand, F_feature, F_int):
        super(GuidedAttention, self).__init__()
        self.W_hand = nn.Sequential(
            nn.Conv2d(F_hand, F_int, kernel_size=1, stride=1, padding=0,
                      bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.W_feature = nn.Sequential(
            nn.Conv2d(F_feature, F_int, kernel_size=1, stride=1, padding=0,
                      bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, hand_x, x):
        psi = self.psi(self.relu(self.W_hand(hand_x) + self.W_feature(x)))

        return x * psi

