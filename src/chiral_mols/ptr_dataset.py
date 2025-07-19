import torch
from torch.utils.data import Dataset
import numpy as np
from typing import List
from chiral_mols.sample import Sample
from chiral_mols.structure_id import StructureID
from pathlib import Path

class PtrMoleculeDataset(Dataset):
    """Stores all atoms in flat tensors and a *ptr* array marking boundaries."""

    def __init__(
        self,
        structure_ids: List[StructureID],
        embeddings: torch.Tensor,  # (N, F)
        labels: torch.Tensor,  # (N,)
        ptr: torch.Tensor,  # (M+1,)
    ) -> None:
        super().__init__()

        if embeddings.shape[0] != labels.shape[0]:
            raise ValueError("Embeddings and labels must have the same #atoms")
        if ptr[0].item() != 0 or ptr[-1].item() != embeddings.shape[0]:
            raise ValueError("ptr[0] must be 0 and ptr[-1] must equal #atoms")
        if len(structure_ids) != ptr.numel() - 1:
            raise ValueError("structure_ids length must equal #molecules")
        
        self.embeddings = embeddings
        self.labels = labels
        self.ptr = ptr
        self.structure_ids = structure_ids


    def __len__(self) -> int:  # number of molecules
        return self.ptr.numel() - 1

    def __getitem__(self, idx: int) -> Sample:
        start, end = self.ptr[idx].item(), self.ptr[idx + 1].item()
        return Sample(
            embeddings=self.embeddings[start:end],  # (n_i, F)
            chirality_labels=self.labels[start:end],  # (n_i,)
        )

    # Convenience ----------------------------------------------------------
    def molecule_slice(self, idx: int) -> slice:
        """Return ``slice`` covering molecule *idx* in the flat tensors."""
        return slice(self.ptr[idx].item(), self.ptr[idx + 1].item())

    @classmethod
    def from_dataset_list(
        cls,
        all_structure_ids: List[StructureID],  
        all_embeddings: List[np.ndarray],  # per-molecule (n_i, F)
        all_chirality_labels: List[List[int]],  # per-molecule (n_i,)
    ) -> "PtrMoleculeDataset":

        ptr = [0]
        emb_tensors = []
        label_tensors = []

        for emb_np, lbl_np in zip(all_embeddings, all_chirality_labels):
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

        return cls(all_structure_ids, embeddings, labels, ptr_tensor)

    def store_dataset(self, dir: Path) -> None:
        dir.mkdir(parents=True, exist_ok=True)
        np.save(dir / "embeddings.npy", self.embeddings.cpu().numpy())
        np.save(dir / "labels.npy", self.labels.cpu().numpy())
        np.save(dir / "ptr.npy", self.ptr.cpu().numpy())
        with open(dir / "structure_ids.txt", "w") as f:
            for sid in self.structure_ids:
                f.write(sid.to_string() + "\n")

    @classmethod
    def reload_dataset_from_dir(cls, dir: Path) -> "PtrMoleculeDataset":
        # load as NumPy, then copy to ensure writeable array
        emb_arr = np.load(dir / "embeddings.npy", mmap_mode="r").copy()
        lbl_arr = np.load(dir / "labels.npy", mmap_mode="r").copy()
        ptr_arr = np.load(dir / "ptr.npy", mmap_mode="r").copy()
        embeddings = torch.from_numpy(emb_arr)
        labels = torch.from_numpy(lbl_arr)
        ptr = torch.from_numpy(ptr_arr)
        with open(dir / "structure_ids.txt", "r") as f:
            structure_ids = [StructureID.from_string(line.strip()) for line in f]
        return cls(structure_ids, embeddings, labels, ptr)
