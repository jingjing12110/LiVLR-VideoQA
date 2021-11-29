# @File : dynamic_rnn.py
# @Github :
import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class PadLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, bidirectional,
                 dropout=0.5):
        super(PadLSTM, self).__init__()

        if num_layers > 1:
            self.rnn = nn.LSTM(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                bidirectional=bidirectional,
                dropout=dropout
            )
        else:
            self.rnn = nn.LSTM(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                bidirectional=bidirectional,
            )

        # self.init_params()

    def init_params(self):
        for i in range(self.rnn.num_layers):
            nn.init.orthogonal_(getattr(self.rnn, f'weight_hh_l{i}'))
            # nn.init.kaiming_normal_(getattr(self.rnn, f'weight_ih_l{i}'))
            nn.init.orthogonal_(getattr(self.rnn, f'weight_ih_l{i}'))
            nn.init.constant_(getattr(self.rnn, f'bias_hh_l{i}'), val=0)
            nn.init.constant_(getattr(self.rnn, f'bias_ih_l{i}'), val=0)
            getattr(self.rnn, f'bias_hh_l{i}').chunk(4)[1].fill_(1)

            if self.rnn.bidirectional:
                nn.init.orthogonal_(
                    getattr(self.rnn, f'weight_hh_l{i}_reverse'))
                # nn.init.kaiming_normal_(
                #     getattr(self.rnn, f'weight_ih_l{i}_reverse'))
                nn.init.orthogonal_(
                    getattr(self.rnn, f'weight_ih_l{i}_reverse'))
                nn.init.constant_(
                    getattr(self.rnn, f'bias_hh_l{i}_reverse'), val=0)
                nn.init.constant_(
                    getattr(self.rnn, f'bias_ih_l{i}_reverse'), val=0)
                getattr(self.rnn, f'bias_hh_l{i}_reverse').chunk(4)[1].fill_(1)

    def forward(self, x, lengths):
        """
        @param x: [seq_len, batch_size, input_size]
        @param lengths: [batch_size]
        @return:
        """
        packed_x = pack_padded_sequence(x, lengths, batch_first=False,
                                        enforce_sorted=False)
        # packed_x, packed_output: PackedSequence
        # hidden: [num_layers * bidirectional, batch_size, hidden_size]
        # cell: [num_layers * bidirectional, batch_size, hidden_size]
        self.rnn.flatten_parameters()
        packed_output, (hidden, cell) = self.rnn(packed_x)

        # output: [real_seq_len, batch_size, hidden_size * 2]
        # output_lengths: [batch_size]
        output, output_lengths = pad_packed_sequence(packed_output)

        return output, (hidden, cell)


class DynamicRNN(nn.Module):
    def __init__(self, rnn_model):
        super().__init__()
        self.rnn_model = rnn_model

    def forward(self, seq_input, seq_lens, initial_state=None):
        """A wrapper over pytorch's rnn to handle sequences of variable length.
        Arguments
        ---------
        seq_input : torch.Tensor
            Input sequence tensor (padded) for RNN model.
            Shape: (batch_size, max_sequence_length, embed_size)
        seq_lens : torch.LongTensor
            Length of sequences (b, )
        initial_state : torch.Tensor
            Initial (hidden, cell) states of RNN model.

        Returns
        -------
            A single tensor of shape (batch_size, rnn_hidden_size) corresponding
            to the outputs of the RNN model at the last time step of each input
            sequence.
        """
        # print('=====================seq_input shape',seq_input.shape)
        max_sequence_length = seq_input.size(1)
        sorted_len, fwd_order, bwd_order = self._get_sorted_order(seq_lens)
        sorted_seq_input = seq_input.index_select(0, fwd_order)
        packed_seq_input = pack_padded_sequence(
            sorted_seq_input, lengths=sorted_len, batch_first=True)

        if initial_state is not None:
            hx = initial_state
            sorted_hx = [x.index_select(1, fwd_order) for x in hx]
            assert hx[0].size(0) == self.rnn_model.num_layers
        else:
            hx = None

        self.rnn_model.flatten_parameters()
        outputs, (h_n, c_n) = self.rnn_model(packed_seq_input, hx)

        # pick hidden and cell states of last layer
        h_n = h_n[-1].index_select(dim=0, index=bwd_order)
        c_n = c_n[-1].index_select(dim=0, index=bwd_order)

        outputs = pad_packed_sequence(
            outputs, batch_first=True, total_length=max_sequence_length
        )
        return outputs, (h_n, c_n)

    @staticmethod
    def _get_sorted_order(lens):
        sorted_len, fwd_order = torch.sort(lens.contiguous().view(-1), 0,
                                           descending=True)
        _, bwd_order = torch.sort(fwd_order)
        sorted_len = list(sorted_len)
        return sorted_len, fwd_order, bwd_order


class DynamicGRU(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1, bias=True,
                 batch_first=False, dropout=0, bidirectional=False):
        super().__init__()
        self.batch_first = batch_first
        self.gru = nn.GRU(input_size, hidden_size, num_layers=num_layers,
                          bias=bias, batch_first=batch_first, dropout=dropout,
                          bidirectional=bidirectional)
        self.gru.flatten_parameters()

    def forward(self, x, seq_len, max_num_frames):
        sorted_seq_len, sorted_idx = torch.sort(seq_len, dim=0, descending=True)
        _, original_idx = torch.sort(sorted_idx, dim=0, descending=False)
        if self.batch_first:
            sorted_x = x.index_select(0, sorted_idx)
        else:
            sorted_x = x.index_select(1, sorted_idx)

        packed_x = nn.utils.rnn.pack_padded_sequence(
            sorted_x, sorted_seq_len.cpu().data.numpy(),
            batch_first=self.batch_first)

        out, state = self.gru(packed_x)

        unpacked_x, unpacked_len = nn.utils.rnn.pad_packed_sequence(out,
                                                                    batch_first=self.batch_first)

        if self.batch_first:
            out = unpacked_x.index_select(0, original_idx)
            if out.shape[1] < max_num_frames:
                out = F.pad(out, [0, 0, 0, max_num_frames - out.shape[1]])
        else:
            out = unpacked_x.index_select(1, original_idx)
            if out.shape[0] < max_num_frames:
                out = F.pad(out, [0, 0, 0, 0, 0, max_num_frames - out.shape[0]])

        # state = state.transpose(0, 1).contiguous().view(out.size(0), -1)
        return out


if __name__ == '__main__':
    rnn = nn.LSTM(512, 512, 2, batch_first=True, dropout=0.1)
    rnn_dynamic = DynamicRNN(rnn)
    # _, (h, c) = rnn_dynamic()


