import copy
import math

import torch
from torch import nn
from torch.nn import functional as F
from torch_scatter import scatter_add

from torchdrug import core, tasks, layers, models, data, metrics
from torchdrug.layers import functional
from torchdrug.core import Registry as R


@R.register("tasks.ResidueTypePrediction")
class ResidueTypePrediction(tasks.AttributeMasking, core.Configurable):

    def predict_and_target(self, batch, all_loss=None, metric=None):
        graph = batch["graph"]
        if self.graph_construction_model:
            graph = self.graph_construction_model.apply_node_layer(graph)

        num_nodes = graph.num_residues if graph.view == "residue" else graph.num_nodes
        num_cum_nodes = num_nodes.cumsum(0)
        num_samples = (num_nodes * self.mask_rate).long().clamp(1)
        num_sample = num_samples.sum()
        sample2graph = torch.repeat_interleave(num_samples)
        node_index = (torch.rand(num_sample, device=self.device) * num_nodes[sample2graph]).long()
        node_index = node_index + (num_cum_nodes - num_nodes)[sample2graph]

        target = graph.residue_type[node_index]
        mask_id = self.model.sequence_model.alphabet.get_idx("<mask>")
        with graph.residue():
            graph.residue_feature[node_index] = 0
            graph.residue_type[node_index] = mask_id
        if self.graph_construction_model:
           graph = self.graph_construction_model.apply_edge_layer(graph)
        input = graph.residue_feature.float()

        output = self.model(graph, input, all_loss, metric)
        node_feature = output["node_feature"][node_index]
        pred = self.mlp(node_feature)

        return pred, target
    

@R.register("tasks.MSP")
class MSP(tasks.InteractionPrediction):

    def __init__(self, model, task, num_mlp_layer=1, graph_construction_model=None, verbose=0):
        super(MSP, self).__init__(model, model2=model, task=task, criterion="bce",
            metric=("auroc", "auprc"), num_mlp_layer=num_mlp_layer, normalization=False,
            num_class=1, graph_construction_model=graph_construction_model, verbose=verbose)

    def preprocess(self, train_set, valid_set, test_set):
        weight = []
        for task, w in self.task.items():
            weight.append(w)

        self.register_buffer("weight", torch.as_tensor(weight, dtype=torch.float))
        self.num_class = [1]
        hidden_dims = [self.model.output_dim] * (self.num_mlp_layer - 1)
        self.mlp = layers.MLP(self.model.output_dim + self.model2.output_dim, hidden_dims + [sum(self.num_class)])  

    def predict(self, batch, all_loss=None, metric=None):
        graph1 = batch["graph1"]
        if self.graph_construction_model:
            graph1 = self.graph_construction_model(graph1)
        graph2 = batch["graph2"]
        if self.graph_construction_model:
            graph2 = self.graph_construction_model(graph2)
        output1 = self.model(graph1, graph1.node_feature.float(), all_loss=all_loss, metric=metric)
        output2 = self.model2(graph2, graph2.node_feature.float(), all_loss=all_loss, metric=metric)
        assert graph1.num_residue == graph2.num_residue
        residue_mask = graph1.residue_type != graph2.residue_type
        node_mask1 = residue_mask[graph1.atom2residue].float().unsqueeze(-1)
        output1 = scatter_add(output1["node_feature"] * node_mask1, graph1.atom2graph, dim=0, dim_size=graph1.batch_size) \
                / (scatter_add(node_mask1, graph1.atom2graph, dim=0, dim_size=graph1.batch_size) + 1e-10)
        node_mask2 = residue_mask[graph2.atom2residue].float().unsqueeze(-1)
        output2 = scatter_add(output2["node_feature"] * node_mask2, graph2.atom2graph, dim=0, dim_size=graph2.batch_size) \
                / (scatter_add(node_mask2, graph2.atom2graph, dim=0, dim_size=graph2.batch_size) + 1e-10)
        pred = self.mlp(torch.cat([output1, output2], dim=-1))
        return pred