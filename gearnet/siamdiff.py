import os
import glob
import math
import numpy as np

import torch
from torch.nn import functional as F
from torch_scatter import scatter_sum, scatter_mean, scatter_add

from torchdrug import core, tasks, layers, models, metrics, data
from torchdrug.data import constant
from torchdrug.layers import functional
from torchdrug.core import Registry as R



@R.register("transforms.NoiseTransform")
class NoiseTransform(core.Configurable):

    def __init__(self, sigma=0.3):
        self.sigma = sigma

    def __call__(self, item):
        graph = item["graph"].clone()
        perturb_noise = torch.randn_like(graph.node_position)
        graph.node_position = graph.node_position + perturb_noise * self.sigma
        item["graph2"] = graph
        return item
    

@R.register("tasks.SiamDiff")
class SiamDiff(tasks.Task, core.Configurable):

    """
    Siamese Diffusion Trajectory Prediction.

    Parameters:
        model (nn.Module): the protein structure encoder to be pre-trained
        sigma_begin (float): the smallest noise scale
        sigma_end (float): the largest noise scale
        num_noise_level (int): the number of noise scale levels
        gamma (float, optional): controls the weights between sequence and structure denoising; 
            (1 - gamma) * seq_loss + gamma * struct_loss
        max_mask_ratio (float, optional): the maximum masking ratio in sequence diffusion
        num_mlp_layer (int, optional): the number of MLP layers for prediction head
        graph_construction_model (nn.Module, optional): graph construction model
        use_MI (bool, optional): whether to use mutual information maximization; if True, use SiamDiff; otherwise, use DiffPreT
    """

    num_class = constant.NUM_AMINO_ACID
    min_mask_ratio = 0.15
    eps = 1e-10

    def __init__(self, model, sigma_begin, sigma_end, num_noise_level, use_MI=True, gamma=0.5,
                max_mask_ratio=1.0, num_mlp_layer=2, graph_construction_model=None):
        super(SiamDiff, self).__init__()
        self.model = model
        self.num_noise_level = num_noise_level
        self.max_mask_ratio = max_mask_ratio
        self.use_MI = use_MI
        self.gamma = gamma
        betas = torch.linspace(-6, 6, num_noise_level)
        betas = betas.sigmoid() * (sigma_end - sigma_begin) + sigma_begin
        alphas = (1. - betas).cumprod(dim=0)
        self.register_buffer("alphas", alphas)
        self.graph_construction_model = graph_construction_model
        
        output_dim = model.output_dim
        self.struct_mlp = layers.MLP(2 * output_dim, [output_dim] * (num_mlp_layer - 1) + [1])
        self.dist_mlp = layers.MLP(1, [output_dim] * (num_mlp_layer - 1) + [output_dim])
        self.seq_mlp = layers.MLP(output_dim, [output_dim] * (num_mlp_layer - 1) + [self.num_class])

    def add_seq_noise(self, graph, noise_level):
        num_nodes = graph.num_residues
        num_cum_nodes = num_nodes.cumsum(0)
        # decide the mask rate according to the noise level
        mask_rate = (self.max_mask_ratio - self.min_mask_ratio) * ((noise_level + 1) / self.num_noise_level) + self.min_mask_ratio
        num_samples = (num_nodes * mask_rate).long().clamp(1)
        num_sample = num_samples.sum()
        sample2graph = torch.repeat_interleave(num_samples)
        node_index = (torch.rand(num_sample, device=self.device) * num_nodes[sample2graph]).long()
        node_index = node_index + (num_cum_nodes - num_nodes)[sample2graph]
        node_index = node_index.clamp(max=num_cum_nodes[-1]-1)

        seq_target = graph.residue_type[node_index].clone()
        selected_residue = torch.zeros((graph.num_residue,), dtype=torch.bool, device=graph.device)
        selected_residue[node_index] = 1
        # only keep backbone atoms of the selected residues
        node_mask = (graph.atom_name == graph.atom_name2id["CA"]) \
                  | (graph.atom_name == graph.atom_name2id["C"]) \
                  | (graph.atom_name == graph.atom_name2id["N"]) \
                  | ~selected_residue[graph.atom2residue]
        
        return node_index, node_mask, seq_target

    def add_struct_noise(self, graph, noise_level):
        # add noise to coordinates and change the pairwise distance in edge features if neccessary
        a_graph = self.alphas[noise_level]      # (num_graph,)
        a_pos = a_graph[graph.node2graph]
        a_edge = a_graph[graph.edge2graph]
        node_in, node_out = graph.edge_list.t()[:2]
        dist = (graph.node_position[node_in] - graph.node_position[node_out]).norm(dim=-1)
        perturb_noise = torch.randn_like(graph.node_position)
        graph.node_position = graph.node_position + perturb_noise * ((1.0 - a_pos).sqrt() / a_pos.sqrt()).unsqueeze(-1)
        perturbed_dist = (graph.node_position[node_in] - graph.node_position[node_out]).norm(dim=-1)
        if self.graph_construction_model and self.graph_construction_model.edge_feature == "gearnet":
            graph.edge_feature[:, -1] = perturbed_dist
        struct_target = (dist - perturbed_dist) / (1.0 - a_edge).sqrt() * a_edge.sqrt()

        return graph, perturbed_dist, struct_target

    def eq_transform(self, score_d, graph):
        # transform invariant scores on edges to equivariant coordinates on nodes
        node_in, node_out = graph.edge_list.t()[:2]
        diff = graph.node_position[node_in] - graph.node_position[node_out]
        dd_dr = diff / (diff.norm(dim=-1, keepdim=True) + 1e-10)
        score_pos = scatter_mean(dd_dr * score_d.unsqueeze(-1), node_in, dim=0, dim_size=graph.num_node) \
                + scatter_mean(- dd_dr * score_d.unsqueeze(-1), node_out, dim=0, dim_size=graph.num_node)
        return score_pos

    def struct_predict(self, output, graph, perturbed_dist):
        # predict scores on edges with node representations and perturbed distance
        node_in, node_out = graph.edge_list.t()[:2]
        dist_pred = self.dist_mlp(perturbed_dist.unsqueeze(-1))
        edge_pred = self.struct_mlp(torch.cat((output[node_in] * output[node_out], dist_pred), dim=-1))
        pred = edge_pred.squeeze(-1)
        return pred

    def seq_predict(self, graph, output, node_index):
        node_feature = scatter_mean(output, graph.atom2residue, dim=0, dim_size=graph.num_residue)[node_index]
        seq_pred = self.seq_mlp(node_feature)
        return seq_pred

    def predict_and_target(self, batch, all_loss=None, metric=None):
        graph1 = batch["graph"]
        if self.graph_construction_model:
            graph1 = self.graph_construction_model.apply_node_layer(graph1)
        
        if self.use_MI:
            graph2 = batch["graph2"]
            graph2.view = graph1.view
            if self.graph_construction_model:
                graph2 = self.graph_construction_model.apply_node_layer(graph2) 

        noise_level = torch.randint(0, self.alphas.shape[0], (graph1.batch_size,), device=self.device) # (num_graph, )
        # add shared sequence noise
        if self.gamma < 1.0:
            node_index, node_mask, seq_target = self.add_seq_noise(graph1, noise_level)
            if graph1.view in ["node", "atom"]:
                graph1.atom_feature[node_mask, -21:] = 0
            graph1 = graph1.subgraph(node_mask)
            with graph1.residue():
                graph1.residue_feature[node_index] = 0
                graph1.residue_type[node_index] = 0
            if self.use_MI:
                if graph2.view in ["node", "atom"]:
                    graph2.atom_feature[node_mask, -21:] = 0
                graph2 = graph2.subgraph(node_mask)
                with graph2.residue():
                    graph2.residue_feature[node_index] = 0
                    graph2.residue_type[node_index] = 0

        # construct edges and edge features
        # this should be done before structure perturbation
        if self.graph_construction_model:
            graph1 = self.graph_construction_model.apply_edge_layer(graph1)
        if self.use_MI and self.graph_construction_model:
            graph2 = self.graph_construction_model.apply_edge_layer(graph2)

        # add structure noise
        if self.gamma > 0.0:
            graph1, perturbed_dist1, struct_target1 = self.add_struct_noise(graph1, noise_level)
            if self.use_MI:
                graph2, perturbed_dist2, struct_target2 = self.add_struct_noise(graph2, noise_level)
        
        output1 = self.model(graph1, graph1.node_feature.float(), all_loss, metric)["node_feature"]
        if self.use_MI:
            output2 = self.model(graph2, graph2.node_feature.float(), all_loss, metric)["node_feature"]
        else:
            output2 = output1

        # predict structure noise
        if self.gamma > 0.0:
            # get invariant scores on edges instead of nodes
            # following https://github.com/MinkaiXu/GeoDiff/blob/ea0ca48045a2f7abfccd7f0df449e45eb6eae638/models/epsnet/dualenc.py#L305-L308
            struct_pred1 = self.struct_predict(output2, graph1, perturbed_dist1)
            struct_pred1 = self.eq_transform(struct_pred1, graph1)
            struct_target1 = self.eq_transform(struct_target1, graph1)
            loss1 = 0.5 * ((struct_pred1 - struct_target1) ** 2).sum(dim=-1) 
            loss1 = scatter_mean(loss1, graph1.node2graph, dim=0, dim_size=graph1.batch_size).mean()
            if self.use_MI:
                struct_pred2 = self.struct_predict(output1, graph2, perturbed_dist2)
                struct_pred2 = self.eq_transform(struct_pred2, graph2)
                struct_target2 = self.eq_transform(struct_target2, graph2)

                loss2 = 0.5 * ((struct_pred2 - struct_target2) ** 2).sum(dim=-1) 
                loss2 = scatter_mean(loss2, graph2.node2graph, dim=0, dim_size=graph2.batch_size).mean()
            else:
                loss2 = loss1
            metric["structure denoising loss"] = loss1 + loss2
            all_loss += self.gamma * (loss1 + loss2)
            pred, target = struct_pred1, struct_target1

        # predict sequence noise
        if self.gamma < 1.0:
            seq_pred1 = self.seq_predict(graph1, output2, node_index)
            loss1 = 0.5 * F.cross_entropy(seq_pred1, seq_target, reduction="none")
            loss1 = scatter_mean(loss1, graph1.residue2graph[node_index], dim=0, dim_size=graph1.batch_size).mean()
            acc1 = (seq_pred1.argmax(dim=-1) == seq_target).float().mean()
            if self.use_MI:
                seq_pred2 = self.seq_predict(graph2, output1, node_index)
                loss2 = 0.5 * F.cross_entropy(seq_pred2, seq_target, reduction="none")
                loss2 = scatter_mean(loss2, graph2.residue2graph[node_index], dim=0, dim_size=graph2.batch_size).mean()
        
                acc2 = (seq_pred2.argmax(dim=-1) == seq_target).float().mean()
            else:
                loss2 = loss1
                acc2 = acc1
            metric["sequence denoising accuracy"] = 0.5 * (acc1 + acc2)
            metric["sequence denoising loss"] = loss1 + loss2
            all_loss += (1 - self.gamma) * (loss1 + loss2)
            pred, target = seq_pred1, seq_target

        metric["loss"] = all_loss

        return pred, target

    def forward(self, batch):
        """"""
        all_loss = torch.tensor(0, dtype=torch.float32, device=self.device)
        metric = {}

        pred, target = self.predict_and_target(batch, all_loss, metric)
        
        return all_loss, metric