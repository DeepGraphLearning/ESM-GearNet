from collections.abc import Sequence
import math
import copy

import torch
from torch import nn
from torch.nn import functional as F

from torch_scatter import scatter_add

from torchdrug import core, layers, data
from torchdrug.layers import functional
from torchdrug.core import Registry as R

from torch_scatter import scatter_mean, scatter_add
from torch_cluster import radius_graph


@R.register("transforms.ResidueGraph")
class ResidueGraph(core.Configurable):

    def __init__(self, keys="graph"):
        if isinstance(keys, str):
            keys = [keys]
        self.keys = keys

    def ca_graph(self, graph):
        mask = (graph.atom_name == data.Protein.atom_name2id["CA"]) & (graph.atom2residue != -1)
        residue2num_atom = graph.atom2residue[mask].bincount(minlength=graph.num_residue)
        residue_mask = residue2num_atom > 0
        mask = mask & residue_mask[graph.atom2residue]
        graph = graph.subgraph(mask).subresidue(residue_mask)
        assert (graph.num_node == graph.num_residue).all()

        return graph

    def __call__(self, item):
        item = item.copy()
        for key in self.keys:
            graph = self.ca_graph(item[key])
            item[key] = graph
        return item
    
    
@R.register("transforms.Orientation")
class Orientation(core.Configurable):

    def __init__(self, keys="graph"):
        if isinstance(keys, str):
            keys = [keys]
        self.keys = keys

    def orientation(self, pos):
        u = F.normalize(pos[1:,:] - pos[:-1,:], p=2, dim=1)
        u1 = u[1:,:]
        u2 = u[:-1, :]
        b = F.normalize(u2 - u1, p=2, dim=1)
        n = F.normalize(torch.cross(u2, u1), p=2, dim=1)
        o = F.normalize(torch.cross(b, n), p=2, dim=1)
        ori = torch.stack([b, n, o], dim=1)
        return torch.cat([ori[0].unsqueeze(0), ori, ori[-1].unsqueeze(0)], dim=0)

    def __call__(self, item):
        new_item = item.copy()
        for key in self.keys:
            graph = item[key]
            if graph.num_residue > 0:
                with graph.residue():
                    graph.orientation = self.orientation(graph.node_position)
            else:
                with graph.residue():
                    graph.orientation = torch.zeros((0, 3, 3))

            new_item[key] = graph
        return new_item
    

def kaiming_uniform(tensor, size):
    fan = 1
    for i in range(1, len(size)):
        fan *= size[i]
    gain = math.sqrt(2.0 / (1 + math.sqrt(5) ** 2))
    std = gain / math.sqrt(fan)
    bound = math.sqrt(3.0) * std  # Calculate uniform bounds from standard deviation
    with torch.no_grad():
        return tensor.uniform_(-bound, bound)
    

class Linear(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 batch_norm: bool = True,
                 dropout: float = 0.0,
                 bias: bool = False,
                 leakyrelu_negative_slope: float = 0.1,
                 momentum: float = 0.2) -> nn.Module:
        super(Linear, self).__init__()

        module = []
        if batch_norm:
            module.append(nn.BatchNorm1d(in_channels, momentum=momentum))
        module.append(nn.LeakyReLU(leakyrelu_negative_slope))
        module.append(nn.Dropout(dropout))
        module.append(nn.Linear(in_channels, out_channels, bias = bias))
        self.module = nn.Sequential(*module)

    def forward(self, x):
        return self.module(x)
    

class MLP(nn.Module):
    def __init__(self,
                 in_channels: int,
                 mid_channels: int,
                 out_channels: int,
                 batch_norm: bool,
                 layer_norm: bool = False,
                 dropout: float = 0.0,
                 bias: bool = True,
                 leakyrelu_negative_slope: float = 0.2,
                 momentum: float = 0.2) -> nn.Module:
        super(MLP, self).__init__()

        module = []
        if batch_norm:
            module.append(nn.BatchNorm1d(in_channels, momentum=momentum))
        if layer_norm:
            module.append(nn.LayerNorm(in_channels))
        module.append(nn.LeakyReLU(leakyrelu_negative_slope))
        module.append(nn.Dropout(dropout))
        if mid_channels is None:
            module.append(nn.Linear(in_channels, out_channels, bias = bias))
        else:
            module.append(nn.Linear(in_channels, mid_channels, bias = bias))
        if batch_norm:
            if mid_channels is None:
                module.append(nn.BatchNorm1d(out_channels, momentum=momentum))
            else:
                module.append(nn.BatchNorm1d(mid_channels, momentum=momentum))
        if layer_norm:
            if mid_channels is None:
                module.append(nn.LayerNorm(out_channels))
            else:
                module.append(nn.LayerNorm(mid_channels))
        module.append(nn.LeakyReLU(leakyrelu_negative_slope))
        if mid_channels is None:
            module.append(nn.Dropout(dropout))
        else:
            module.append(nn.Linear(mid_channels, out_channels, bias = bias))

        self.module = nn.Sequential(*module)

    def forward(self, input):
        return self.module(input)
        

class WeightNet(nn.Module):
    def __init__(self, l, kernel_channels):
        super(WeightNet, self).__init__()

        self.l = l
        self.kernel_channels = kernel_channels

        self.Ws = nn.ParameterList()
        self.bs = nn.ParameterList()

        for i, channels in enumerate(kernel_channels):
            if i == 0:
                self.Ws.append(torch.nn.Parameter(torch.empty(l, 3 + 3 + 1, channels)))
                self.bs.append(torch.nn.Parameter(torch.empty(l, channels)))
            else:
                self.Ws.append(torch.nn.Parameter(torch.empty(l, kernel_channels[i-1], channels)))
                self.bs.append(torch.nn.Parameter(torch.empty(l, channels)))

        self.relu = nn.LeakyReLU(0.2)

    def reset_parameters(self):
        for i, channels in enumerate(self.kernel_channels):
            if i == 0:
                kaiming_uniform(self.Ws[0].data, size=[self.l, 3 + 3 + 1, channels])
            else:
                kaiming_uniform(self.Ws[i].data, size=[self.l, self.kernel_channels[i-1], channels])
            self.bs[i].data.fill_(0.0)

    def forward(self, input, idx):
        for i in range(len(self.kernel_channels)):
            W = torch.index_select(self.Ws[i], 0, idx)
            b = torch.index_select(self.bs[i], 0, idx)
            if i == 0:
                weight = self.relu(torch.bmm(input.unsqueeze(1), W).squeeze(1) + b)
            else:
                weight = self.relu(torch.bmm(weight.unsqueeze(1), W).squeeze(1) + b)

        return weight
    

class CDConv(layers.MessagePassingBase):

    def __init__(self, input_dim, output_dim, kernel_dims, seq_cutoff, spatial_cutoff):
        super(CDConv, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.kernel_dims = kernel_dims
        self.seq_cutoff = seq_cutoff
        self.spatial_cutoff = spatial_cutoff

        self.WeightNet = WeightNet(seq_cutoff, kernel_dims)
        self.W = torch.nn.Parameter(torch.empty(kernel_dims[-1] * input_dim, output_dim))

        self.reset_parameters()

    def reset_parameters(self):
        self.WeightNet.reset_parameters()
        kaiming_uniform(self.W.data, size=[self.kernel_dims * self.input_dim, self.output_dim])

    def message(self, graph, input):
        node_in, node_out = graph.edge_list.t()[:2]

        ori = graph.orientation
        t = graph.node_position[node_in] - graph.node_position[node_out]
        dist = t.norm(dim=-1, keepdim=True)
        t /= (dist + 1e-9)
        t = torch.matmul(ori[node_out], t.unsqueeze(2)).squeeze(2)
        r = torch.sum(ori[node_out] * ori[node_in], dim=-1)
        normed_distance = dist / self.spatial_cutoff

        seq_dist = graph.residue_number[node_in] - graph.residue_number[node_out]
        s = self.seq_cutoff // 2
        seq_dist = torch.clamp(input=seq_dist, min=-s, max=s)
        seq_idx = (seq_dist + s).long()
        normed_length = torch.abs(seq_dist).unsqueeze(-1) / s

        delta = torch.cat([t, r, dist], dim=-1)
        kernel_weight = self.WeightNet(delta, seq_idx)

        smooth = 0.5 - torch.tanh(normed_distance*normed_length*16.0 - 14.0)*0.5

        message = torch.matmul((kernel_weight*smooth).unsqueeze(2), input[node_in].unsqueeze(1))
        message = message.flatten(-2, -1)
        return message

    def aggregate(self, graph, message):
        node_out = graph.edge_list[:, 1]
        update = scatter_add(message, node_out, dim=0, dim_size=graph.num_node)
        return update

    def combine(self, input, update):
        update = torch.matmul(update, self.W)
        return update
    

class CDBlock(nn.Module):
    def __init__(self,
                 l: float,
                 r: float,
                 kernel_dims,
                 input_dim: int,
                 output_dim: int,
                 base_width: float = 16.0,
                 batch_norm: bool = True,
                 dropout: float = 0.0,
                 bias: bool = False,
                 leakyrelu_negative_slope: float = 0.1,
                 ) -> nn.Module:

        super(CDBlock, self).__init__()

        if input_dim != output_dim:
            self.identity = Linear(in_channels=input_dim,
                                  out_channels=output_dim,
                                  batch_norm=batch_norm,
                                  dropout=dropout,
                                  bias=bias,
                                  leakyrelu_negative_slope=leakyrelu_negative_slope)
        else:
            self.identity = nn.Sequential()

        width = int(output_dim * (base_width / 64.))
        self.input = MLP(in_channels=input_dim,
                         mid_channels=None,
                         out_channels=width,
                         batch_norm=batch_norm,
                         dropout=dropout,
                         bias=bias,
                         leakyrelu_negative_slope=leakyrelu_negative_slope)
        self.conv = CDConv(seq_cutoff=l, spatial_cutoff=r, kernel_dims=kernel_dims, input_dim=width, output_dim=width)
        self.output = Linear(in_channels=width,
                             out_channels=output_dim,
                             batch_norm=batch_norm,
                             dropout=dropout,
                             bias=bias,
                             leakyrelu_negative_slope=leakyrelu_negative_slope)

    def forward(self, graph, x):
        identity = self.identity(x)
        x = self.input(x)
        x = self.conv(graph, x)
        out = self.output(x) + identity
        return out
    

@R.register("models.CDConv")
class ContinuousDiscreteConvolutionalNetwork(nn.Module, core.Configurable):

    def __init__(self, input_dim, embedding_dim, hidden_dims, cutoffs, seq_cutoff, kernel_dims, base_width=16.0,
                 batch_norm=True, dropout=0.2, bias=False):
        super(ContinuousDiscreteConvolutionalNetwork, self).__init__()

        if not isinstance(hidden_dims, Sequence):
            hidden_dims = [hidden_dims]
        self.input_dim = input_dim
        self.output_dim = hidden_dims[-1]
        self.dims = [embedding_dim] + list(hidden_dims)
        self.cutoffs = cutoffs
        self.seq_cutoff = seq_cutoff

        self.embedding = nn.Linear(input_dim, embedding_dim, bias=False)
        self.layers = nn.ModuleList()
        for i in range(len(self.dims) - 1):
            self.layers.append(CDBlock(seq_cutoff, cutoffs[i], kernel_dims, self.dims[i], self.dims[i+1], base_width=base_width,
                                     batch_norm=batch_norm, dropout=dropout, bias=bias))
            self.layers.append(CDBlock(seq_cutoff, cutoffs[i], kernel_dims, self.dims[i+1], self.dims[i+1], base_width=base_width,
                                     batch_norm=batch_norm, dropout=dropout, bias=bias))

        self.readout = layers.MeanReadout()

    def receptive_field(self, graph, cutoff):
        edge_list = radius_graph(graph.node_position, r=cutoff, batch=graph.node2graph, max_num_neighbors=9999) # (2, E)
        edge2graph = graph.node2graph[edge_list[0]]
        num_edges = scatter_add(torch.ones_like(edge2graph), edge2graph, dim_size=graph.batch_size)
        edge_list, num_edges = functional._extend(
            edge_list.t(),
            num_edges,
            torch.stack([
                torch.arange(graph.num_node, dtype=torch.long, device=graph.device),
                torch.arange(graph.num_node, dtype=torch.long, device=graph.device)
            ], dim=1),
            graph.num_nodes
        )
        offsets = (graph.num_cum_nodes - graph.num_nodes).repeat_interleave(num_edges)

        new_graph = data.PackedGraph(edge_list, num_nodes=graph.num_nodes, num_edges=num_edges,
                                    offsets=offsets)
        with new_graph.node():
            new_graph.node_position = graph.node_position
            new_graph.residue_number = graph.residue_number
            new_graph.orientation = graph.orientation
        
        return new_graph

    def pooling(self, graph, input):
        num_nodes = torch.div(graph.num_nodes + 1, 2, rounding_mode="floor")
        old2new = functional.variadic_arange(graph.num_nodes)
        old2new = torch.div(old2new, 2, rounding_mode="floor")
        old2new = old2new + (num_nodes.cumsum(0) - num_nodes)[graph.node2graph]

        output = scatter_mean(input, old2new, dim=0, dim_size=num_nodes.sum())

        new2old = functional.variadic_arange(num_nodes) * 2
        node2graph = torch.arange(graph.batch_size, device=graph.device).repeat_interleave(num_nodes)
        new2old = new2old + (graph.num_nodes.cumsum(0) - graph.num_nodes)[node2graph]

        new_graph = graph.subgraph(new2old)
        with new_graph.node():
            new_graph.node_position = scatter_mean(graph.node_position, old2new, dim=0, dim_size=num_nodes.sum())
            new_graph.residue_number = torch.div(graph.residue_number[new2old], 2, rounding_mode='floor')
            ori = scatter_mean(src=graph.orientation, index=old2new, dim=0, dim_size=num_nodes.sum())
            new_graph.orientation = F.normalize(ori, dim=-1)

        return new_graph, output

    def forward(self, graph, input, all_loss=None, metric=None):
        x = self.embedding(input)
        ori_node2graph = graph.node2graph

        for i in range(len(self.layers) // 2):
            graph = self.receptive_field(graph, self.cutoffs[i])
            x = self.layers[2*i](graph, x)
            x = self.layers[2*i+1](graph, x)
            if i != len(self.layers) // 2 - 1:
                graph, x = self.pooling(graph, x)

        graph_feature = self.readout(graph, x)

        return {
            "node_feature": graph_feature[ori_node2graph],
            "graph_feature": graph_feature,
        }