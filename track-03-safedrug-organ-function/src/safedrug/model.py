"""
SafeDrugModel ported from SOTA/SafeDrug/src/models.py (ycq091044/SafeDrug,
master branch), with one architectural change: an optional organ-function
vector is concatenated into the LAST visit's patient representation before
the query projection (design spec §3). When organ_dim=0 the model is
architecturally IDENTICAL to the original SafeDrug (the baseline arm of
the ablation); when organ_dim=73 it is the +OrganFunction arm. Both arms
use this same class, differing only in construction and input shape.

Note on `self.mpnn_emb` and `retain_graph`: `self.mpnn_emb` is computed once in
`__init__` from a throwaway `MolecularGraphNeuralNetwork` and its autograd graph
stays attached for the lifetime of the model; every `forward()` call reuses that
same graph. A caller that calls `loss.backward()` more than once across training
steps (as any standard training loop does) must call
`loss.backward(retain_graph=True)`, or the second call will raise "Trying to
backward through the graph a second time". This is faithful to upstream
SafeDrug, which relies on its trainer doing the same.
"""
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter


class MaskLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, mask):
        weight = torch.mul(self.weight, mask)
        output = torch.mm(input, weight)
        if self.bias is not None:
            return output + self.bias
        return output


class MolecularGraphNeuralNetwork(nn.Module):
    def __init__(self, n_fingerprint, dim, layer_hidden, device):
        super().__init__()
        self.device = device
        self.embed_fingerprint = nn.Embedding(n_fingerprint, dim).to(self.device)
        self.W_fingerprint = nn.ModuleList(
            [nn.Linear(dim, dim).to(self.device) for _ in range(layer_hidden)]
        )
        self.layer_hidden = layer_hidden

    def pad(self, matrices, pad_value):
        shapes = [m.shape for m in matrices]
        M, N = sum(s[0] for s in shapes), sum(s[1] for s in shapes)
        zeros = torch.FloatTensor(np.zeros((M, N))).to(self.device)
        pad_matrices = pad_value + zeros
        i, j = 0, 0
        for k, matrix in enumerate(matrices):
            m, n = shapes[k]
            pad_matrices[i : i + m, j : j + n] = matrix
            i += m
            j += n
        return pad_matrices

    def update(self, matrix, vectors, layer):
        hidden_vectors = torch.relu(self.W_fingerprint[layer](vectors))
        return hidden_vectors + torch.mm(matrix, hidden_vectors)

    def sum(self, vectors, axis):
        sum_vectors = [torch.sum(v, 0) for v in torch.split(vectors, axis)]
        return torch.stack(sum_vectors)

    def forward(self, inputs):
        fingerprints, adjacencies, molecular_sizes = inputs
        fingerprints = torch.cat(fingerprints)
        adjacencies = self.pad(adjacencies, 0)
        fingerprint_vectors = self.embed_fingerprint(fingerprints)
        for layer in range(self.layer_hidden):
            hs = self.update(adjacencies, fingerprint_vectors, layer)
            fingerprint_vectors = hs
        return self.sum(fingerprint_vectors, molecular_sizes)


class SafeDrugModel(nn.Module):
    def __init__(
        self,
        vocab_size,
        ddi_adj,
        ddi_mask_h,
        mpnn_set,
        n_fingerprint,
        average_projection,
        emb_dim=64,
        organ_dim=0,
        device=torch.device("cpu"),
    ):
        super().__init__()
        self.device = device
        self.organ_dim = organ_dim

        self.embeddings = nn.ModuleList(
            [nn.Embedding(vocab_size[i], emb_dim) for i in range(2)]
        )
        self.dropout = nn.Dropout(p=0.5)
        self.encoders = nn.ModuleList(
            [nn.GRU(emb_dim, emb_dim, batch_first=True) for _ in range(2)]
        )
        self.query = nn.Sequential(
            nn.ReLU(), nn.Linear(2 * emb_dim + organ_dim, emb_dim)
        )

        self.bipartite_transform = nn.Sequential(
            nn.Linear(emb_dim, ddi_mask_h.shape[1])
        )
        self.bipartite_output = MaskLinear(ddi_mask_h.shape[1], vocab_size[2], False)

        self.mpnn_molecule_set = list(zip(*mpnn_set))
        self.mpnn_emb = MolecularGraphNeuralNetwork(
            n_fingerprint, emb_dim, layer_hidden=2, device=device
        ).forward(self.mpnn_molecule_set)
        self.mpnn_emb = torch.mm(
            average_projection.to(device=self.device),
            self.mpnn_emb.to(device=self.device),
        )
        self.mpnn_emb.to(device=self.device)
        self.mpnn_output = nn.Linear(vocab_size[2], vocab_size[2])
        self.mpnn_layernorm = nn.LayerNorm(vocab_size[2])

        self.tensor_ddi_adj = torch.FloatTensor(ddi_adj).to(device)
        self.tensor_ddi_mask_h = torch.FloatTensor(ddi_mask_h).to(device)
        self.init_weights()

    def forward(self, input):
        i1_seq, i2_seq = [], []

        def sum_embedding(embedding):
            return embedding.sum(dim=1).unsqueeze(dim=0)

        for adm in input:
            i1 = sum_embedding(
                self.dropout(
                    self.embeddings[0](
                        torch.LongTensor(adm[0]).unsqueeze(dim=0).to(self.device)
                    )
                )
            )
            i2 = sum_embedding(
                self.dropout(
                    self.embeddings[1](
                        torch.LongTensor(adm[1]).unsqueeze(dim=0).to(self.device)
                    )
                )
            )
            i1_seq.append(i1)
            i2_seq.append(i2)
        i1_seq = torch.cat(i1_seq, dim=1)
        i2_seq = torch.cat(i2_seq, dim=1)

        o1, _ = self.encoders[0](i1_seq)
        o2, _ = self.encoders[1](i2_seq)
        patient_representations = torch.cat([o1, o2], dim=-1).squeeze(dim=0)
        last_repr = patient_representations[-1:, :]

        if self.organ_dim > 0:
            organ_vec = torch.FloatTensor(input[-1][3]).unsqueeze(dim=0).to(self.device)
            last_repr = torch.cat([last_repr, organ_vec], dim=-1)

        query = self.query(last_repr)

        mpnn_match = F.sigmoid(torch.mm(query, self.mpnn_emb.t()))
        mpnn_att = self.mpnn_layernorm(mpnn_match + self.mpnn_output(mpnn_match))

        bipartite_emb = self.bipartite_output(
            F.sigmoid(self.bipartite_transform(query)), self.tensor_ddi_mask_h.t()
        )

        result = torch.mul(bipartite_emb, mpnn_att)

        neg_pred_prob = F.sigmoid(result)
        neg_pred_prob = neg_pred_prob.t() * neg_pred_prob
        batch_neg = 0.0005 * neg_pred_prob.mul(self.tensor_ddi_adj).sum()

        return result, batch_neg

    def init_weights(self):
        initrange = 0.1
        for item in self.embeddings:
            item.weight.data.uniform_(-initrange, initrange)
