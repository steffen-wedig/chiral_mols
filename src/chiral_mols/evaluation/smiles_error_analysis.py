from rdkit import Chem
from collections import defaultdict
import numpy as np 
def min_diff_distance(mol, center_idx, max_depth=10):
    # build neighbor‐sequences for each substituent
    seqs = []
    for nbr in mol.GetAtomWithIdx(center_idx).GetNeighbors():
        seq = []
        current = [nbr.GetIdx()]
        for depth in range(max_depth):
            # gather atomic numbers at this depth
            atomic_nums = [mol.GetAtomWithIdx(i).GetAtomicNum() for i in current]
            seq.append(tuple(sorted(atomic_nums)))  # sort to be order‐independent
            # advance one more hop
            next_atoms = []
            for i in current:
                for nn in mol.GetAtomWithIdx(i).GetNeighbors():
                    if nn.GetIdx() != center_idx and nn.GetIdx() not in current:
                        next_atoms.append(nn.GetIdx())
            current = list(set(next_atoms))
        seqs.append(seq)
    # find minimal depth where seqs differ
    for d in range(len(seqs[0])):
        depths = {seq[d] for seq in seqs}
        if len(depths) > 1:
            return d+1
    return None  # no difference found within max_depth


def analyse_smiles_list_cip_distance(smiles_list : list[str]):

    results = []
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        chiral_centers = Chem.FindMolChiralCenters(mol, includeUnassigned=True)
        if len(chiral_centers)==1:
            idx,_ = chiral_centers[0]
            d = min_diff_distance(mol, idx)
            results.append(d)


    return results