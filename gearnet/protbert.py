import os

import torch
from torch import nn
from transformers import BertModel, BertTokenizer

from torchdrug import core, layers, utils, data
from torchdrug.layers import functional
from torchdrug.core import Registry as R


@R.register("models.ProtBert")
class ProtBert(nn.Module, core.Configurable):
    """
    The protein language model, ProtBert-BFD proposed in
    `ProtTrans: Towards Cracking the Language of Life's Code Through Self-Supervised Deep Learning and High Performance Computing`_.

    .. _ProtTrans: Towards Cracking the Language of Life's Code Through Self-Supervised Deep Learning and High Performance Computing:
        https://arxiv.org/ftp/arxiv/papers/2007/2007.06225.pdf

    Parameters:
        path (str): path to store ProtBert model weights.
        readout (str, optional): readout function. Available functions are ``pooler``, ``sum`` and ``mean``.
    """

    url = "https://zenodo.org/record/4633647/files/prot_bert_bfd.zip"
    md5 = "30fad832a088eb879e0ff88fa70c9655"
    output_dim = 1024

    def __init__(self, path, readout="pooler"):
        super(ProtBert, self).__init__()
        path = os.path.expanduser(path)
        if not os.path.exists(path):
            os.makedirs(path)
        self.path = path

        model, tokenizer = self.load_weight(path)
        mapping = self.construct_mapping(tokenizer)
        self.model = model
        self.tokenizer = tokenizer
        self.register_buffer("mapping", mapping)

        if readout == "pooler":
            self.readout = None
        elif readout == "sum":
            self.readout = layers.SumReadout("residue")
        elif readout == "mean":
            self.readout = layers.MeanReadout("residue")
        else:
            raise ValueError("Unknown readout `%s`" % readout)

    def load_weight(self, path):
        zip_file = utils.download(self.url, path, md5=self.md5)
        model_path = os.path.join(utils.extract(zip_file), 'prot_bert_bfd')
        tokenizer = BertTokenizer.from_pretrained(model_path, do_lower_case=False)
        model = BertModel.from_pretrained(model_path)
        return model, tokenizer

    def construct_mapping(self, tokenizer):
        mapping = [0] * len(data.Protein.id2residue_symbol)
        for i, token in data.Protein.id2residue_symbol.items():
            mapping[i] = tokenizer._convert_token_to_id(token)
        mapping = torch.tensor(mapping)
        return mapping

    def forward(self, graph, input, all_loss=None, metric=None):
        input = graph.residue_type
        input = self.mapping[input]
        size = graph.num_residues
        size_ext = size
        bos = torch.ones(graph.batch_size, dtype=torch.long, device=self.device) * self.tokenizer.cls_token_id
        input, size_ext = functional._extend(bos, torch.ones_like(size_ext), input, size_ext)
        eos = torch.ones(graph.batch_size, dtype=torch.long, device=self.device) * self.tokenizer.sep_token_id
        input, size_ext = functional._extend(input, size_ext, eos, torch.ones_like(size_ext))
        input = functional.variadic_to_padded(input, size_ext, value=self.tokenizer.pad_token_id)[0]

        output = self.model(input)
        residue_feature = output.last_hidden_state
        graph_feature = output.pooler_output

        residue_feature = functional.padded_to_variadic(residue_feature, size_ext)
        starts = size_ext.cumsum(0) - size_ext
        starts = starts + 1
        ends = starts + size
        mask = functional.multi_slice_mask(starts, ends, len(residue_feature))
        residue_feature = residue_feature[mask]
        if self.readout:
            graph_feature = self.readout(graph, residue_feature)

        return {
            "graph_feature": graph_feature,
            "residue_feature": residue_feature
        }