from chiral_mols.data.structure_id import StructureID
from typing import List, Sequence
import numpy as np

class DatasetSplitter():

    def __init__(self, structure_ids: List[StructureID]):
        
        self.structure_ids = structure_ids

    def random_split_by_molecule(
        self,
        train_val_ratios: Sequence[float] = (0.7, 0.3),
        shuffle: bool = True
    ) -> List[List[int]]:
        """
        Split structure_ids into N folds by MoleculeID, where N = len(train_val_ratios).

        Args:
            train_val_ratios: sequence of floats, one per desired split. 
                Theyll be normalized to sum to 1.
            shuffle: if True, the moleculeIDs are first randomly permuted.

        Returns:
            A list of length N; each element is a list of StructureID belonging
            to that split.
        """
        # 1. Normalize ratios
        ratios = np.array(train_val_ratios, dtype=float)
        if ratios.ndim != 1 or (ratios <= 0).any():
            raise ValueError("train_val_ratios must be a 1D sequence of positive floats")
        ratios = ratios / ratios.sum()

        # 2. Get unique molecule IDs
        all_mol_ids = np.array([sid.MoleculeID for sid in self.structure_ids])
        unique_mol_ids = np.unique(all_mol_ids)

        # 3. Optionally shuffle
        if shuffle:
            perm = np.random.permutation(len(unique_mol_ids))
            unique_mol_ids = unique_mol_ids[perm]

        # 4. Compute split indices
        n = len(unique_mol_ids)
        # cumulative counts at which to cut
        cut_points = (np.cumsum(ratios) * n).astype(int)
        # ensure last cut equals n
        cut_points[-1] = n

        # 5. Slice molecule ID groups and collect
        splits: List[List[StructureID]] = []
        start = 0
        for end in cut_points:
            group = set(unique_mol_ids[start:end])
            subset = [i for i, sid in enumerate(self.structure_ids) if sid.MoleculeID in group]
            splits.append(subset)
            start = end

        return splits
