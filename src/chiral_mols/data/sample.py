from dataclasses import dataclass
import torch
from typing import Sequence


@dataclass
class Sample:
    embeddings: torch.Tensor
    chirality_labels: torch.Tensor
    atom_mask: torch.Tensor | None = None

    def to_(self, *, device: torch.device | str) -> "Sample":
        """In-place device migration for *all* tensors (dtype unchanged)."""
        self.embeddings = self.embeddings.to(device=device)
        self.chirality_labels = self.chirality_labels.to(device=device)
        if self.atom_mask is not None:
            self.atom_mask = self.atom_mask.to(device=device)
        return self


def ptr_collate_padding(batch: Sequence[Sample]) -> Sample:
    """Pad a *list of samples* into a single ``Sample`` holding the whole batch.

    Returned shapes::
        embeddings       - (B, max_n, F)
        chirality_labels - (B, max_n)
        atom_mask        - (B, max_n)  (True where an atom is present)


    """
    B = len(batch)
    n_atoms = torch.tensor([s.embeddings.shape[0] for s in batch], dtype=torch.long)
    max_n = int(n_atoms.max())

    D_dim = batch[0].embeddings.shape[1]

    # Allocate padded tensors ------------------------------------------------
    emb_pad = batch[0].embeddings.new_zeros((B, max_n, D_dim))
    label_pad = batch[0].chirality_labels.new_zeros((B, max_n))
    mask = torch.zeros((B, max_n), dtype=torch.bool)

    for i, sample in enumerate(batch):
        n = sample.embeddings.shape[0]
        emb_pad[i, :n] = sample.embeddings
        label_pad[i, :n] = sample.chirality_labels
        mask[i, :n] = True  # Mask is true for real atoms

    return Sample(embeddings=emb_pad, chirality_labels=label_pad, atom_mask=mask)


@dataclass
class FlatSample:
    embeddings: torch.Tensor  # (N_total, F)
    chirality_labels: torch.Tensor  # (N_total,)
    mol_idx: torch.Tensor  # (N_total,) tells you which molecule each atom came from

    def to_(self, *, device: torch.device | str) -> "FlatSample":
        self.embeddings = self.embeddings.to(device)
        self.chirality_labels = self.chirality_labels.to(device)
        self.mol_idx = self.mol_idx.to(device)
        return self


def concat_collate(batch: Sequence[Sample]) -> FlatSample:
    """Collate `Sample`s by *concatenating* atoms instead of padding."""
    # How many atoms per molecule?
    lengths = torch.tensor([s.embeddings.size(0) for s in batch], dtype=torch.long)

    # Build the flat tensors
    emb = torch.cat([s.embeddings for s in batch], dim=0)  # (N_total, F)
    lbl = torch.cat([s.chirality_labels for s in batch], dim=0)  # (N_total,)

    # Optional: which molecule does each atom belong to?
    mol_idx = torch.repeat_interleave(torch.arange(len(batch)), lengths)  # (N_total,)

    return FlatSample(embeddings=emb, chirality_labels=lbl, mol_idx=mol_idx)
