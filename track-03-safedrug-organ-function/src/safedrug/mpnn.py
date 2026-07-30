"""
Ports SafeDrug's buildMPNN (SOTA/SafeDrug/src/util.py) so this project owns
the code rather than importing across to a sibling project's uncommitted
path at runtime. Produces the per-molecule (fingerprints, adjacency,
molecular_size) tuples MolecularGraphNeuralNetwork (model.py) consumes,
plus the averaging-projection matrix that maps molecule-level embeddings
back to one row per med_voc ATC3 code (zero row for codes with no SMILES).
"""
from collections import defaultdict

import numpy as np
import torch
from rdkit import Chem


def _create_atoms(mol, atom_dict):
    atoms = [a.GetSymbol() for a in mol.GetAtoms()]
    for a in mol.GetAromaticAtoms():
        i = a.GetIdx()
        atoms[i] = (atoms[i], "aromatic")
    atoms = [atom_dict[a] for a in atoms]
    return np.array(atoms)


def _create_ijbonddict(mol, bond_dict):
    i_jbond_dict = defaultdict(lambda: [])
    for b in mol.GetBonds():
        i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        bond = bond_dict[str(b.GetBondType())]
        i_jbond_dict[i].append((j, bond))
        i_jbond_dict[j].append((i, bond))
    return i_jbond_dict


def _extract_fingerprints(radius, atoms, i_jbond_dict, fingerprint_dict, edge_dict):
    if (len(atoms) == 1) or (radius == 0):
        nodes = [fingerprint_dict[a] for a in atoms]
    else:
        nodes = atoms
        i_jedge_dict = i_jbond_dict
        for _ in range(radius):
            nodes_ = []
            for i, j_edge in i_jedge_dict.items():
                neighbors = [(nodes[j], edge) for j, edge in j_edge]
                fingerprint = (nodes[i], tuple(sorted(neighbors)))
                nodes_.append(fingerprint_dict[fingerprint])
            i_jedge_dict_ = defaultdict(lambda: [])
            for i, j_edge in i_jedge_dict.items():
                for j, edge in j_edge:
                    both_side = tuple(sorted((nodes[i], nodes[j])))
                    edge = edge_dict[(both_side, edge)]
                    i_jedge_dict_[i].append((j, edge))
            nodes = nodes_
            i_jedge_dict = i_jedge_dict_
    return np.array(nodes)


def build_mpnn_set(molecule, med_voc_idx2word: dict, radius: int, device):
    """molecule: safedrug.ddi_mask.build_molecule_map()'s output.
    med_voc_idx2word: voc_final2.pkl's med_voc.idx2word.
    Returns (mpnn_set, n_fingerprint, average_projection):
    - mpnn_set: list of (fingerprints: LongTensor, adjacency: FloatTensor, molecular_size: int)
      tuples, one per resolvable SMILES across all codes.
    - n_fingerprint: int, size for MolecularGraphNeuralNetwork's embedding table.
    - average_projection: FloatTensor (len(med_voc_idx2word), len(mpnn_set)) -
      row i averages the embeddings of code i's own molecule instances;
      an all-zero row means code i had no resolvable SMILES."""
    atom_dict = defaultdict(lambda: len(atom_dict))
    bond_dict = defaultdict(lambda: len(bond_dict))
    fingerprint_dict = defaultdict(lambda: len(fingerprint_dict))
    edge_dict = defaultdict(lambda: len(edge_dict))
    mpnn_set = []
    average_index = []

    for _, atc3 in med_voc_idx2word.items():
        counter = 0
        for smiles in sorted(molecule[atc3]):  # sorted() required: set iteration order depends on PYTHONHASHSEED (randomized per-process by default)
            try:
                mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
                atoms = _create_atoms(mol, atom_dict)
                molecular_size = len(atoms)
                i_jbond_dict = _create_ijbonddict(mol, bond_dict)
                fingerprints = _extract_fingerprints(
                    radius, atoms, i_jbond_dict, fingerprint_dict, edge_dict
                )
                adjacency = Chem.GetAdjacencyMatrix(mol)
                for _ in range(adjacency.shape[0] - fingerprints.shape[0]):
                    fingerprints = np.append(fingerprints, 1)

                fingerprints = torch.LongTensor(fingerprints).to(device)
                adjacency = torch.FloatTensor(adjacency).to(device)
                mpnn_set.append((fingerprints, adjacency, molecular_size))
                counter += 1
            except Exception:
                continue
        average_index.append(counter)

    n_fingerprint = len(fingerprint_dict)
    n_col = sum(average_index)
    n_row = len(average_index)

    average_projection = np.zeros((n_row, n_col))
    col = 0
    for i, count in enumerate(average_index):
        if count > 0:
            average_projection[i, col : col + count] = 1 / count
        col += count

    return mpnn_set, n_fingerprint, torch.FloatTensor(average_projection)
