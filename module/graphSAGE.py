# @File : graphSAGE.py 
# @Github : https://github.com/MrLeeeee/GCN-GAT-and-Graphsage

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class GraphSAGELayer(nn.Module):
    def __init__(self, dim_input, dim_hidden):
        super(GraphSAGELayer, self).__init__()
        self.dim_input = dim_input
        self.W = nn.Parameter(torch.zeros(size=(2 * dim_input, dim_hidden)),
                              requires_grad=True)
        self.bias = nn.Parameter(torch.zeros(dim_hidden), requires_grad=True)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.W.size(1))
        self.W.data.uniform_(-stdv, stdv)
        self.bias.data.uniform_(-stdv, stdv)

    def forward(self, x, adj):
        """
        @param x: [n_node, dim_feat]
        @param adj: [n_node, n_node]
        @return:
        """
        h1 = torch.mm(adj, x)
        degree = adj.sum(axis=1).repeat(self.dim_input, 1).T
        h1 = h1 / degree
        h1 = torch.cat([x, h1], dim=1)
        h1 = torch.mm(h1, self.W)

        return h1


class GraphSAGE(nn.Module):
    def __init__(self, nfeat, nhid, nclass):
        super(GraphSAGE, self).__init__()
        self.dropout = nn.Dropout()
        self.sage1 = GraphSAGELayer(nfeat, nhid)
        self.sage2 = GraphSAGELayer(nhid, nhid)
        self.att = nn.Linear(nhid, nclass)

    def forward(self, x, adj):
        hid1 = self.sage1(x, adj)
        hid1 = self.dropout(hid1)
        hid2 = self.sage2(hid1, adj)
        out = self.att(hid2)

        return F.log_softmax(out, dim=1)


# ##############################################################################
# Inductive Representation Learning on Large Graphs
# GraphSAGE: 
# ##############################################################################
class Aggregator(nn.Module):
    def __init__(self, input_dim=None, output_dim=None, device='cpu'):
        """
        Parameters
        ----------
        input_dim : int or None.
            Dimension of input node features. Used for defining fully
            connected layer in pooling aggregators. Default: None.
        output_dim : int or None
            Dimension of output node features. Used for defining fully
            connected layer in pooling aggregators. Currently only works when
            input_dim = output_dim. Default: None.
        """
        super(Aggregator, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.device = device

    def forward(self, features, nodes, mapping, rows, num_samples=25):
        """
        Parameters
        ----------
        features : torch.Tensor
            An (n' x input_dim) tensor of input node features.
        nodes : numpy array
            nodes is a numpy array of nodes in the current layer of
            the computation graph.
        mapping : dict
            mapping is a dictionary mapping node v (labelled 0 to |V|-1) to
            its position in the layer of nodes in the computation graph
            before nodes. For example, if the layer before nodes is [2,5],
            then mapping[2] = 0 and mapping[5] = 1.
        rows : numpy array
            rows[i] is an array of neighbors of node i which is present in nodes.
        num_samples : int
            Number of neighbors to sample while aggregating. Default: 25.
        Returns
        -------
        out : torch.Tensor
            An (len(nodes) x output_dim) tensor of output node features.
            Currently only works when output_dim = input_dim.
        """
        _choice, _len, _min = np.random.choice, len, min
        mapped_rows = [np.array([mapping[v] for v in row], dtype=np.int64) for row
                       in rows]
        if num_samples == -1:
            sampled_rows = mapped_rows
        else:
            sampled_rows = [_choice(row, _min(_len(row), num_samples),
                                    _len(row) < num_samples) for row in
                            mapped_rows]

        n = _len(nodes)
        if self.__class__.__name__ == 'LSTMAggregator':
            out = torch.zeros(n, 2 * self.output_dim).to(self.device)
        else:
            out = torch.zeros(n, self.output_dim).to(self.device)
        for i in range(n):
            if _len(sampled_rows[i]) != 0:
                out[i, :] = self._aggregate(features[sampled_rows[i], :])

        return out

    def _aggregate(self, features):
        raise NotImplementedError


class MeanAggregator(Aggregator):
    def _aggregate(self, features):
        return torch.mean(features, dim=0)


class PoolAggregator(Aggregator):
    def __init__(self, input_dim, output_dim, device='cpu'):
        super(PoolAggregator, self).__init__(input_dim, output_dim, device)

        self.fc1 = nn.Linear(input_dim, output_dim)
        self.relu = nn.ReLU()

    def _aggregate(self, features):
        out = self.relu(self.fc1(features))
        return self._pool_fn(out)

    def _pool_fn(self, features):
        raise NotImplementedError


class MaxPoolAggregator(PoolAggregator):
    def _pool_fn(self, features):
        return torch.max(features, dim=0)[0]


class MeanPoolAggregator(PoolAggregator):
    def _pool_fn(self, features):
        return torch.mean(features, dim=0)


class LSTMAggregator(Aggregator):
    def __init__(self, input_dim, output_dim, device='cpu'):
        super().__init__(input_dim, output_dim, device)

        self.lstm = nn.LSTM(input_dim, output_dim, bidirectional=True,
                            batch_first=True)

    def _aggregate(self, features):
        perm = np.random.permutation(np.arange(features.shape[0]))
        features = features[perm, :]
        features = features.unsqueeze(0)

        out, _ = self.lstm(features)
        out = out.squeeze(0)
        out = torch.sum(out, dim=0)

        return out


class VanillaGraphSAGE(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim,
                 agg_class=MaxPoolAggregator, dropout=0.5,
                 num_samples=25, device='cpu'):
        """
        Parameters
        ----------
        input_dim : int
            Dimension of input node features.
        hidden_dims : list of ints
            Dimension of hidden layers. Must be non empty.
        output_dim : int
            Dimension of output node features.
        agg_class : An aggregator class.
            Aggregator. One of the aggregator classes imported at the top of
            this module. Default: MaxPoolAggregator.
        dropout : float
            Dropout rate. Default: 0.5.
        num_samples : int
            Number of neighbors to sample while aggregating. Default: 25.
        device : str
            'cpu' or 'cuda:0'. Default: 'cpu'.
        """
        super(VanillaGraphSAGE, self).__init__()
        self.num_samples = num_samples
        self.num_layers = len(hidden_dims) + 1

        self.aggregators = nn.ModuleList(
            [agg_class(input_dim, input_dim, device)])
        self.aggregators.extend(
            [agg_class(dim, dim, device) for dim in hidden_dims])

        c = 3 if agg_class == LSTMAggregator else 2
        self.fcs = nn.ModuleList([nn.Linear(c * input_dim, hidden_dims[0])])
        self.fcs.extend(
            [nn.Linear(c * hidden_dims[i - 1], hidden_dims[i]) for i in
             range(1, len(hidden_dims))])
        self.fcs.extend([nn.Linear(c * hidden_dims[-1], output_dim)])

        self.bns = nn.ModuleList(
            [nn.BatchNorm1d(hidden_dim) for hidden_dim in hidden_dims])

        self.dropout = nn.Dropout(dropout)

        self.relu = nn.ReLU()

    def forward(self, features, node_layers, mappings, rows):
        """
        Parameters
        ----------
        features : torch.Tensor
            An (n' x input_dim) tensor of input node features.
        node_layers : list of numpy array
            node_layers[i] is an array of the nodes in the ith layer of the
            computation graph.
        mappings : list of dictionary
            mappings[i] is a dictionary mapping node v (labelled 0 to |V|-1)
            in node_layers[i] to its position in node_layers[i]. For example,
            if node_layers[i] = [2,5], then mappings[i][2] = 0 and
            mappings[i][5] = 1.
        rows : numpy array
            rows[i] is an array of neighbors of node i.
        Returns
        -------
        out : torch.Tensor
            An (len(node_layers[-1]) x output_dim) tensor of output node features.
        """
        out = features
        for k in range(self.num_layers):
            nodes = node_layers[k + 1]
            mapping = mappings[k]
            init_mapped_nodes = np.array([mappings[0][v] for v in nodes],
                                         dtype=np.int64)
            cur_rows = rows[init_mapped_nodes]
            aggregate = self.aggregators[k](out, nodes, mapping, cur_rows,
                                            self.num_samples)
            cur_mapped_nodes = np.array([mapping[v] for v in nodes],
                                        dtype=np.int64)
            out = torch.cat((out[cur_mapped_nodes, :], aggregate), dim=1)
            out = self.fcs[k](out)
            if k + 1 < self.num_layers:
                out = self.relu(out)
                out = self.bns[k](out)
                out = self.dropout(out)
                out = out.div(out.norm(dim=1, keepdim=True) + 1e-6)

        return out
