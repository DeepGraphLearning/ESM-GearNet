import os
import math
import h5py
import glob
import random
import logging
import warnings
from tqdm import tqdm

import numpy as np
import pandas as pd
import scipy
from sklearn.preprocessing import normalize

import torch
from torch.utils import data as torch_data
from torch.utils.data import IterableDataset

from torchdrug import data, utils, core, datasets
from torchdrug.layers import functional
from torchdrug.core import Registry as R
from torchdrug.utils import comm

from atom3d import datasets as da
from atom3d.datasets import LMDBDataset


logger = logging.getLogger(__name__)


class Atom3DDataset:

    def protein_from_data_frame(self, df, atom_feature=None, bond_feature=None, 
                                residue_feature="default", mol_feature=None):  
        assert bond_feature is None
        assert mol_feature is None
        atom_feature = data.Protein._standarize_option(atom_feature)
        bond_feature = data.Protein._standarize_option(bond_feature)
        mol_feature = data.Protein._standarize_option(mol_feature)
        residue_feature = data.Protein._standarize_option(residue_feature)
        
        atom2residue = []
        atom_type = []
        residue_type = []
        atom_name = []
        is_hetero_atom = []
        residue_number = []
        occupancy = []
        b_factor = []
        insertion_code = []
        chain_id = []
        node_position = []
        _residue_feature = []
        _atom_feature = []
        last_residue = None
        for i, atom in df.iterrows():
            atom_type.append(data.feature.atom_vocab.get(atom['element'], 0))
            type = atom['resname']
            number = atom['residue']
            code = atom['insertion_code']
            canonical_residue = (number, code, type)
            if canonical_residue != last_residue:
                last_residue = canonical_residue
                if type not in data.Protein.residue2id:
                    warnings.warn("Unknown residue `%s`. Treat as glycine" % type)
                    type = "GLY"
                residue_type.append(data.Protein.residue2id[type])
                residue_number.append(number)
                insertion_code.append(data.Protein.alphabet2id.get(code, 0))
                chain_id.append(data.Protein.alphabet2id.get(atom['chain'], 0))
                feature = []
                for name in residue_feature:
                    if name == "default":
                        feature = data.feature.onehot(type, data.feature.residue_vocab, allow_unknown=True)
                    else:
                        raise ValueError('Feature %s not included' % name)
                _residue_feature.append(feature)
            name = atom['name']
            if name not in data.Protein.atom_name2id:
                name = "UNK"
            atom_name.append(data.Protein.atom_name2id[name])
            is_hetero_atom.append(atom['hetero'] != ' ')
            occupancy.append(atom['occupancy'])
            b_factor.append(atom['bfactor'])
            node_position.append([atom['x'], atom['y'], atom['z']])
            atom2residue.append(len(residue_type) - 1)
            feature = []
            for name in atom_feature:
                if name == "residue_symbol":
                    feature += \
                        data.feature.onehot(atom['element'], data.feature.atom_vocab, allow_unknown=True) + \
                        data.feature.onehot(type, data.feature.residue_vocab, allow_unknown=True)
                else:
                    raise ValueError('Feature %s not included' % name)
            _atom_feature.append(feature)
        
        atom_type = torch.tensor(atom_type)
        residue_type = torch.tensor(residue_type)
        atom_name = torch.tensor(atom_name)
        is_hetero_atom = torch.tensor(is_hetero_atom)
        occupancy = torch.tensor(occupancy)
        b_factor = torch.tensor(b_factor)
        atom2residue = torch.tensor(atom2residue)
        residue_number = torch.tensor(residue_number)
        insertion_code = torch.tensor(insertion_code)
        chain_id = torch.tensor(chain_id)
        node_position = torch.tensor(node_position)
        if len(residue_feature) > 0:
            _residue_feature = torch.tensor(_residue_feature)
        else:
            _residue_feature = None
        if len(atom_feature) > 0:
            _atom_feature = torch.tensor(_atom_feature)
        else:
            _atom_feature = None

        return data.Protein(edge_list=None, num_node=len(atom_type), atom_type=atom_type, bond_type=[], 
                    residue_type=residue_type, atom_name=atom_name, atom2residue=atom2residue, 
                    is_hetero_atom=is_hetero_atom, occupancy=occupancy, b_factor=b_factor,
                    residue_number=residue_number, insertion_code=insertion_code, chain_id=chain_id, 
                    node_position=node_position, atom_feature=_atom_feature, residue_feature=_residue_feature)

    @torch.no_grad()
    def construct_graph(self, data_list, model=None, batch_size=1, gpus=None, verbose=True):
        protein_list = []
        if gpus is None:
            device = torch.device("cpu")
        else:
            device = torch.device(gpus[comm.get_rank() % len(gpus)])
        model = model.to(device)
        t = range(0, len(data_list), batch_size)
        if verbose:
            t = tqdm(t, desc="Constructing graphs for training")
        for start in t:
            end = start + batch_size
            batch = data_list[start:end]
            proteins = data.Protein.pack(batch).to(device)
            if gpus and hasattr(proteins, "residue_feature"):
                with proteins.residue():
                    proteins.residue_feature = proteins.residue_feature.to_dense()
            proteins = model(proteins).cpu()
            for protein in proteins:
                if gpus and hasattr(protein, "residue_feature"):
                    with protein.residue():
                        protein.residue_feature = protein.residue_feature.to_sparse()
                protein_list.append(protein)
        return protein_list

    def split(self):
        offset = 0
        splits = []
        for num_sample in self.num_samples:
            split = torch_data.Subset(self, range(offset, offset + num_sample))
            splits.append(split)
            offset += num_sample
        return splits
    

@R.register("datasets.PSRDataset")
class PSRDataset(data.ProteinDataset, Atom3DDataset):

    url = "https://zenodo.org/record/4915648/files/PSR-split-by-year.tar.gz"
    dir_name = "PSR-split-by-year"
    md5 = "8647b9d10d0a79dff81d1d83c825e74c"
    processed_file = "PSR-split-by-year.pkl.gz"

    def __init__(self, path, transform=None, verbose=1, **kwargs):
        path = os.path.join(os.path.expanduser(path), self.dir_name)
        if not os.path.exists(path):
            os.makedirs(path)
        self.path = path

        tar_file = utils.download(self.url, path, md5=self.md5)
        utils.extract(tar_file)
        pkl_file = os.path.join(path, self.processed_file)

        if os.path.exists(pkl_file):
            self.load_pickle(pkl_file, transform=transform, lazy=False, verbose=verbose, **kwargs)
        else:
            self.transform = transform
            self.data = []
            self.sequences = []
            self.pdb_files = []
            self.kwargs = kwargs
            for split in ['train', 'val', 'test']:
                self.load_lmdb(os.path.join(path, 'split-by-year', 'data', split), verbose, **kwargs)
            self.save_pickle(pkl_file, verbose=verbose)
        
        splits = [os.path.basename(os.path.dirname(pdb_file)) for pdb_file in self.pdb_files]
        self.num_samples = [splits.count("train"), splits.count("val"), splits.count("test")]
        
    def load_lmdb(self, lmdb_path, verbose, **kwargs):
        dataset = da.load_dataset(lmdb_path, "lmdb")
        if verbose:
            dataset = tqdm(dataset, "Constructing proteins from data frames")
        for i, data in enumerate(dataset):
            protein = self.protein_from_data_frame(data["atoms"], **kwargs)
            if not protein:
                logger.debug("Can't construct protein from pdb file `%s`. Ignore this sample." % data["id"])
                continue
            if hasattr(protein, "residue_feature"):
                with protein.residue():
                    protein.residue_feature = protein.residue_feature.to_sparse()
            with protein.graph():
                protein.gdt_ts = torch.tensor(data["scores"]["gdt_ts"])
            self.data.append(protein)
            self.sequences.append(protein.to_sequence())
            self.pdb_files.append(os.path.join(lmdb_path, str(i)))

    def get_item(self, index):
        protein = self.data[index]
        protein = protein.subgraph(protein.atom_type != 0)
        
        if hasattr(protein, "residue_feature"):
            with protein.residue():
                protein.residue_feature = protein.residue_feature.to_dense()
        
        item = {"graph": protein}
        item["gdt_ts"] = protein.gdt_ts
        if self.transform:
            item = self.transform(item)
        return item

    @property
    def tasks(self):
        """List of tasks."""
        return ["gdt_ts"]

    def __repr__(self):
        lines = [
            "#sample: %d" % len(self),
            "#task: gdt_ts",
        ]
        return "%s(\n  %s\n)" % (self.__class__.__name__, "\n  ".join(lines))
    

@R.register("transforms.TruncateProteinPair")
class TruncateProteinPair(core.Configurable):

    def __init__(self, max_length=None, random=False):
        self.truncate_length = max_length
        self.random = random

    def __call__(self, item):
        new_item = item.copy()
        graph1 = item["graph1"]
        graph2 = item["graph2"]
        length = graph1.num_residue
        if length <= self.truncate_length:
            return item
        residue_mask = graph1.residue_type != graph2.residue_type
        index = residue_mask.nonzero()[:, 0]
        if self.random:
            start = math.randint(index, min(index + self.truncate_length, length)) - self.truncate_length
        else:
            start = min(index - self.truncate_length // 2, length - self.truncate_length)
        start = max(start, 0)
        end = start + self.truncate_length
        mask = torch.zeros(length, dtype=torch.bool, device=graph1.device)
        mask[start:end] = True
        new_item["graph1"] = graph1.subresidue(mask)
        new_item["graph2"] = graph2.subresidue(mask)

        return new_item
    

@R.register("datasets.MSPDataset")
class MSPDataset(data.ProteinDataset, Atom3DDataset):

    url = "https://zenodo.org/record/4962515/files/MSP-split-by-sequence-identity-30.tar.gz"
    dir_name = "MSP-split-by-sequence-identity-30"
    md5 = "6628e8efac12648d3b78bb0fc0d8860c"
    processed_file = "MSP-split-by-sequence-identity-30.pkl.gz"

    def __init__(self, path, transform=None, verbose=1, **kwargs):
        path = os.path.join(os.path.expanduser(path), self.dir_name)
        if not os.path.exists(path):
            os.makedirs(path)
        self.path = path

        tar_file = utils.download(self.url, path, md5=self.md5)
        utils.extract(tar_file)
        pkl_file = os.path.join(path, self.processed_file)

        if os.path.exists(pkl_file):
            self.load_pickle(pkl_file, transform=transform, lazy=False, verbose=verbose, **kwargs)
        else:
            self.transform = transform
            self.data = []
            self.sequences = []
            self.pdb_files = []
            self.kwargs = kwargs
            for split in ['train', 'val', 'test']:
                self.load_lmdb(os.path.join(path, 'split-by-sequence-identity-30', 'data', split), verbose, **kwargs)
            self.save_pickle(pkl_file, verbose=verbose)

        splits = [os.path.basename(os.path.dirname(pdb_file)) for pdb_file in self.pdb_files]
        self.num_samples = [splits.count("train"), splits.count("val"), splits.count("test")]

    def load_lmdb(self, lmdb_path, verbose, **kwargs):
        dataset = da.load_dataset(lmdb_path, "lmdb")
        if verbose:
            dataset = tqdm(dataset, "Constructing proteins from data frames")
        for i, data in enumerate(dataset):
            wt = self.protein_from_data_frame(data["original_atoms"], **kwargs)
            mt = self.protein_from_data_frame(data["mutated_atoms"], **kwargs)
            if not wt or not mt:
                logger.debug("Can't construct protein from pdb file `%s`. Ignore this sample." % data["id"])
                continue
            if hasattr(wt, "residue_feature"):
                with wt.residue():
                    wt.residue_feature = wt.residue_feature.to_sparse()
                with mt.residue():
                    mt.residue_feature = mt.residue_feature.to_sparse()
            with wt.graph():
                wt.label = torch.tensor(data["label"] == '1')
            self.data.append((wt, mt))
            self.sequences.append((wt.to_sequence(), mt.to_sequence()))
            self.pdb_files.append(os.path.join(lmdb_path, str(i)))

    def get_item(self, index):
        wt = self.data[index][0].clone()
        wt = wt.subgraph(wt.atom_type != 0)
        mt = self.data[index][1].clone()
        mt = mt.subgraph(mt.atom_type != 0)

        if hasattr(wt, "residue_feature"):
            with wt.residue():
                wt.residue_feature = wt.residue_feature.to_dense()
            with mt.residue():
                mt.residue_feature = mt.residue_feature.to_dense()

        item = {"graph1": wt, "graph2": mt}
        item["label"] = wt.label
        if self.transform:
            item = self.transform(item)
        return item

    @property
    def tasks(self):
        """List of tasks."""
        return ["label"]

    def __repr__(self):
        lines = [
            "#sample: %d" % len(self),
            "#task: label",
        ]
        return "%s(\n  %s\n)" % (self.__class__.__name__, "\n  ".join(lines))
    