from dataclasses import dataclass
import torch
from typing import Sequence



@dataclass
class Sample:
    embeddings: torch.Tensor  # (N_total, F)
    chirality_labels: torch.Tensor  # (N_total,)
    structure_id: torch.Tensor  # (N_total,) tells you which molecule each atom came from

    def to_(self, *, device: torch.device | str) -> "Sample":
        self.embeddings = self.embeddings.to(device)
        self.chirality_labels = self.chirality_labels.to(device)
        self.structure_id = self.structure_id.to(device)
        return self


def concat_collate(batch: Sequence[Sample]) -> Sample:
    """Collate `Sample`s by *concatenating* atoms instead of padding."""
    # How many atoms per molecule?
    lengths = torch.tensor([s.embeddings.size(0) for s in batch], dtype=torch.long)

    # Build the flat tensors
    emb = torch.cat([s.embeddings for s in batch], dim=0)  # (N_total, F)
    lbl = torch.cat([s.chirality_labels for s in batch], dim=0)  # (N_total,)
    structure_ids = torch.cat([s.structure_id for s in batch], dim=0)  # (N_total,)
    
    return Sample(embeddings=emb, chirality_labels=lbl, structure_id=structure_ids)
