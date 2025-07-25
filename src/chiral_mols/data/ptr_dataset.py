import torch
from torch.utils.data import Dataset
import numpy as np
from typing import List, Tuple
from chiral_mols.data.sample import Sample
from chiral_mols.data.structure_id import StructureID
from pathlib import Path
from ase import Atoms
from rdkit.Chem import CanonSmiles
from typing import Iterable


class PtrMoleculeDataset(Dataset):
    """Stores all atoms in flat tensors and a *ptr* array marking boundaries."""

    def __init__(
        self,
        structure_ids: List[StructureID],
        embeddings: torch.Tensor,  # (N, F)
        labels: torch.Tensor,  # (N,)
        ptr: torch.Tensor,  # (M+1,)
        positions: torch.Tensor | None,
        atomic_numbers: torch.Tensor | None,
        smiles: list[str],
    ) -> None:
        super().__init__()

        N_atoms = embeddings.shape[0]
        assert labels.shape[0] == N_atoms
        assert ptr[0].item() == 0
        assert ptr[-1].item() == N_atoms
        if atomic_numbers is not None:
            assert atomic_numbers.shape[0] == N_atoms
        if positions is not None:    
            assert positions.shape[0] == N_atoms

        N_mols = len(structure_ids)
        assert len(smiles) == N_mols
        assert N_mols == ptr.numel() - 1

        self.embeddings = embeddings
        self.labels = labels
        self.ptr = ptr
        self.structure_ids = structure_ids

        self.positions = positions
        self.atomic_numbers = atomic_numbers
        self.smiles = smiles

        self.atomwise_structure_ids = self.get_atomwise_structure_ids()

    def __len__(self) -> int:  # number of molecules
        return self.ptr.numel() - 1

    def __getitem__(self, idx: int) -> Sample:
        # This method gets all atoms from a molecule by id

        start, end = self.ptr[idx].item(), self.ptr[idx + 1].item()
        return Sample(
            embeddings=self.embeddings[start:end],  # (n_i, F)
            chirality_labels=self.labels[start:end],
            structure_id=self.atomwise_structure_ids[start:end]  # (n_i,)
        )

    def molecule_slice(self, idx: int) -> slice:
        """Return ``slice`` covering molecule *idx* in the flat tensors."""
        return slice(self.ptr[idx].item(), self.ptr[idx + 1].item())

    def get_atomwise_structure_ids(self):

        assert self.structure_ids is not None

        atomwise_structure_ids = []

        for i, sid in enumerate(self.structure_ids):
            N_atoms = int(self.ptr[i + 1] - self.ptr[i])
            atomwise_structure_ids.extend([sid.StructureID] * N_atoms)

        atomwise_structure_ids = torch.IntTensor(atomwise_structure_ids)

        assert atomwise_structure_ids.shape[0] == len(self.embeddings)
        return atomwise_structure_ids


    def get_atoms_from_structure_id(self, structure_id):

        start = self.ptr[structure_id]
        end = self.ptr[structure_id + 1] 

        atomic_numbers = self.atomic_numbers[start:end]
        atomic_positions = self.positions[start:end, : ]
        smi = self.smiles[structure_id]
        atoms = Atoms(numbers = atomic_numbers, positions= atomic_positions, pbc = False, info = {"smiles" : smi})
        
        return atoms


    @classmethod
    def from_dataset_list(
        cls,
        all_structure_ids: List[StructureID],
        all_embeddings: List[np.ndarray],  # per-molecule (n_i, F)
        all_chirality_labels: List[List[int]],  # per-molecule (n_i,)
        all_atoms: List[Atoms],
    ) -> "PtrMoleculeDataset":

        ptr = [0]
        emb_tensors = []
        label_tensors = []

        for emb_np, lbl_np, sid in zip(
            all_embeddings, all_chirality_labels, all_structure_ids
        ):
            emb_np = np.asarray(emb_np, dtype=np.float32)
            lbl_np = np.asarray(lbl_np, dtype=np.int64)

            if emb_np.shape[0] != lbl_np.shape[0]:
                raise ValueError("Mismatched #atoms between embeddings and labels")

            emb_tensors.append(torch.from_numpy(emb_np))
            label_tensors.append(torch.from_numpy(lbl_np))

            ptr.append(ptr[-1] + emb_np.shape[0])

        embeddings = torch.cat(emb_tensors, dim=0)  # (N, F)
        labels = torch.cat(label_tensors, dim=0)  # (N,)

        ptr_tensor = torch.tensor(ptr, dtype=torch.int64)

        positions, atomic_numbers, smiles = cls.convert_all_atoms(all_atoms)

        return cls(
            all_structure_ids,
            embeddings,
            labels,
            ptr_tensor,
            positions,
            atomic_numbers,
            smiles,
        )

    @staticmethod
    def convert_all_atoms(
        all_atoms: List[Atoms],
    ) -> Tuple[torch.Tensor, torch.Tensor, List[str]]:
        """
        Convert a list of ASE `Atoms` objects into flat tensors.

        Returns
        -------
        positions : torch.FloatTensor, shape (N, 3)
            All atomic Cartesian coordinates concatenated across molecules.
        atomic_numbers : torch.LongTensor, shape (N,)
            All atomic numbers concatenated across molecules.
        """

        pos_list = []
        z_list = []
        smiles = []

        for ats in all_atoms:
            pos_np = np.asarray(ats.get_positions(), dtype=np.float32)  # (n_i, 3)
            z_np = np.asarray(ats.get_atomic_numbers(), dtype=np.int64)  # (n_i,)

            if pos_np.shape[0] != z_np.shape[0]:
                raise ValueError(
                    "Mismatched #atoms between positions and atomic numbers"
                )

            pos_list.append(torch.from_numpy(pos_np))
            z_list.append(torch.from_numpy(z_np))
            
            smi = ats.info.get("smiles")
            if smi is None:
                raise KeyError("Atoms.info['smiles'] missing")
            smiles.append(smi)

        positions = torch.cat(pos_list, dim=0)  # (N, 3)
        atomic_numbers = torch.cat(z_list, dim=0)  # (N,)

        print(f"Atomic Number Shape {atomic_numbers.shape}")

        return positions, atomic_numbers, smiles

    def store_dataset(self, dir: Path) -> None:

        dir.mkdir(parents=True, exist_ok=True)
        np.save(dir / "embeddings.npy", self.embeddings.cpu().numpy())
        np.save(dir / "labels.npy", self.labels.cpu().numpy())
        np.save(dir / "ptr.npy", self.ptr.cpu().numpy())

        if self.positions is not None:
            np.save(dir / "atomic_pos.npy", self.positions.cpu().numpy())
        if self.atomic_numbers is not None: 
            np.save(dir / "atomic_numbers.npy", self.atomic_numbers.cpu().numpy())

        with open(dir / "structure_ids.txt", "w") as f:
            for sid, smi in zip(self.structure_ids, self.smiles):
                f.write(sid.to_string() + " " + smi + "\n")

    @classmethod
    def reload_dataset_from_dir(cls, dir: Path, reload_atoms: bool =  False) -> "PtrMoleculeDataset":
        # load as NumPy, then copy to ensure writeable array
        emb_arr = np.load(dir / "embeddings.npy")
        lbl_arr = np.load(dir / "labels.npy")
        ptr_arr = np.load(dir / "ptr.npy")

        if reload_atoms:
            pos_arr = np.load(dir / "atomic_pos.npy")
            atomic_numbers_arr = np.load(dir / "atomic_numbers.npy")
            positions = torch.from_numpy(pos_arr)
            atomic_numbers = torch.from_numpy(atomic_numbers_arr)
        else: 
            positions = None
            atomic_numbers = None

        embeddings = torch.from_numpy(emb_arr)
        labels = torch.from_numpy(lbl_arr)
        ptr = torch.from_numpy(ptr_arr)
        

        structure_ids = []
        smiles = []
        with open(dir / "structure_ids.txt", "r") as f:
            for line in f:
                line = line.strip()
                sid_str, smi = line.split(" ", 1)
                structure_ids.append(StructureID.from_string(sid_str))
                smiles.append(smi.strip())

        return cls(
            structure_ids,
            embeddings,
            labels,
            ptr,
            positions,
            atomic_numbers,
            smiles=smiles,
        )

    def drop_hydrogens(self) -> "PtrMoleculeDataset":
        assert self.atomic_numbers is not None, "Need atomic_numbers to drop H"

        keep_mask = self.atomic_numbers != 1
        idx_keep = keep_mask.nonzero(as_tuple=False).squeeze(1)

        # per‑molecule kept counts (all > 0 per your assumption)
        mol_lengths = (self.ptr[1:] - self.ptr[:-1]).tolist()
        kept_counts = [m.sum().item() for m in keep_mask.split(mol_lengths)]

        new_ptr = torch.empty_like(self.ptr)
        new_ptr[0] = 0
        new_ptr[1:] = torch.as_tensor(
            kept_counts, dtype=self.ptr.dtype, device=self.ptr.device
        ).cumsum(0)

        return PtrMoleculeDataset(
            structure_ids=self.structure_ids,
            embeddings=self.embeddings[idx_keep],
            labels=self.labels[idx_keep],
            ptr=new_ptr,
            positions=self.positions[idx_keep] if self.positions is not None else None,
            atomic_numbers=self.atomic_numbers[idx_keep],
            smiles=self.smiles,
        )