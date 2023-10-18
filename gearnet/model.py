from collections.abc import Sequence

import torch
from torch import nn
from torch.nn import functional as F

from torch_scatter import scatter_add

from torchdrug import core, layers
from torchdrug.core import Registry as R
from torchdrug.layers import functional


@R.register("models.FusionNetwork")
class FusionNetwork(nn.Module, core.Configurable):

    def __init__(self, sequence_model, structure_model, fusion="series", cross_dim=None):
        super(FusionNetwork, self).__init__()
        self.sequence_model = sequence_model
        self.structure_model = structure_model
        self.fusion = fusion
        if fusion in ["series", "parallel"]:
            self.output_dim = sequence_model.output_dim + structure_model.output_dim
        elif fusion == "cross":
            self.seq_linear = nn.Linear(sequence_model.output_dim, cross_dim)
            self.struct_linear = nn.Linear(structure_model.output_dim, cross_dim)
            self.attn = layers.SelfAttentionBlock(cross_dim, num_heads=8, dropout=0.1)
            self.output_dim = cross_dim * 2            
        else:
            raise ValueError("Not support fusion scheme %s" % fusion)

    def forward(self, graph, input, all_loss=None, metric=None):
        # Sequence model
        output1 = self.sequence_model(graph, input, all_loss, metric)
        node_output1 = output1.get("node_feature", output1.get("residue_feature"))
        # Structure model
        if self.fusion == "series":
            input = node_output1
        output2 = self.structure_model(graph, input, all_loss, metric)
        node_output2 = output2.get("node_feature", output2.get("residue_feature"))
        # Fusion
        if self.fusion in ["series", "parallel"]:
            node_feature = torch.cat([node_output1, node_output2], dim=-1)
            graph_feature = torch.cat([
                output1["graph_feature"], 
                output2["graph_feature"]
            ], dim=-1)
        else:
            seq_output = self.seq_linear(node_output1)
            struct_output = self.struct_linear(node_output2)
            attn_input, sizes = functional._extend(seq_output, graph.num_residues, struct_output, graph.num_residues)
            attn_input, mask = functional.variadic_to_padded(attn_input, sizes)
            attn_output = self.attn(attn_input, mask)
            node_feature = functional.padded_to_variadic(attn_output, sizes)
            seq_index = torch.arange(graph.num_residue, dtype=torch.long, device=graph.device)
            num_cum_residues = torch.cat([torch.zeros((1,), dtype=torch.long, device=graph.device), graph.num_cum_residues])
            seq_index += num_cum_residues[graph.residue2graph]
            struct_index = torch.arange(graph.num_residue, dtype=torch.long, device=graph.device)
            struct_index += graph.num_cum_residues[graph.residue2graph]
            node_feature = torch.cat([node_feature[seq_index], node_feature[struct_index]], dim=-1)
            graph_feature = scatter_add(node_feature, graph.residue2graph, dim=0, dim_size=graph.batch_size)

        return {
            "graph_feature": graph_feature,
            "node_feature": node_feature
        }