from dataclasses import dataclass
import torch
from typing import Sequence

@dataclass
class Sample:
    embeddings: torch.Tensor
    chirality_labels: torch.Tensor
    atom_mask: torch.Tensor | None  = None

    def to_(self, *, device: torch.device | str) -> "Sample":
        """In‑place device migration for *all* tensors (dtype unchanged)."""
        self.embeddings = self.embeddings.to(device=device)
        self.chirality_labels = self.chirality_labels.to(device=device)
        if self.atom_mask is not None:
            self.atom_mask = self.atom_mask.to(device=device)
        return self


def ptr_collate(batch: Sequence[Sample]) -> Sample:  # noqa: D401 (simple func)
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
        mask[i, :n] = True # Mask is true for real atoms

    return Sample(embeddings=emb_pad, chirality_labels=label_pad, atom_mask=mask)