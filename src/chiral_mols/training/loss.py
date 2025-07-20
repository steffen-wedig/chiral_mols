import torch
from collections import Counter
from chiral_mols.data.ptr_dataset import PtrMoleculeDataset
import torch.nn.functional as F
from torch import nn




def get_class_weights(dataset: PtrMoleculeDataset , num_classes: int = 3, scheme: str = "inverse"):
    """
    dataset  – anything that yields .chirality_labels
    scheme   – 'inverse'  : weight = 1 / freq
               'balanced' : weight = N_total / (num_classes * freq)
               'sqrt'     : weight = 1 / sqrt(freq)
    returns  – 1‑D tensor of length num_classes
    """

    chirality_labels = dataset.labels.reshape(-1)
    counts = Counter(int(c) for c in chirality_labels)
    freqs  = torch.tensor([counts.get(i, 0) for i in range(num_classes)],
                          dtype=torch.float)

    if scheme == "inverse":
        w = 1.0 / freqs.clamp(min=1.)                   # avoid div/0
    elif scheme == "balanced":                          # scikit‑learn style
        w = freqs.sum() / (len(freqs) * freqs.clamp(min=1.))
    elif scheme == "sqrt":
        w = 1.0 / torch.sqrt(freqs.clamp(min=1.))
    else:
        raise ValueError("unknown weighting scheme")

    # normalise so that Achiral weight ≈ 1 for easy interpretability
    return w / w[0]





class FocalLoss(nn.Module):
    """
    Multiclass focal loss with soft‑max probabilities
    -------------------------------------------------
    Parameters
    ----------
    gamma : float
        Focusing parameter (γ).  γ = 2 is common.
    alpha : Tensor[ num_classes ] | None
        Per‑class weighting factor (α  ∈  [0,1]).
        Pass the same tensor you used for CrossEntropyLoss weights,
        *or* leave None to disable class weighting.
    reduction : 'mean' | 'sum' | 'none'
        Reduction style (same semantics as nn.CrossEntropyLoss).
    """

    def __init__(self, gamma: float = 2.0, alpha: torch.Tensor | None = None,
                 reduction: str = "mean"):
        super().__init__()
        self.gamma = gamma
        self.reduction = reduction

        if alpha is not None:
            # register as buffer so it moves with .to(device) and is saved in the state‑dict
            self.register_buffer("alpha", alpha.float())
        else:
            self.alpha = None

    def forward(self, logits: torch.Tensor, targets: torch.Tensor):
        """
        logits  : shape (B, C) – raw scores from the network
        targets : shape (B,)   – integer class labels 0 … C‑1
        """
        # --- standard CE loss, per sample -----------------------------------
        ce_loss = F.cross_entropy(logits, targets, reduction="none")   # −log p_t
        p_t = torch.exp(-ce_loss)                                      # p_t

        # --- focal factor ---------------------------------------------------
        focal = (1.0 - p_t) ** self.gamma

        # --- optional alpha‑balancing --------------------------------------
        if self.alpha is not None:
            alpha_t = self.alpha.gather(0, targets)
            focal_loss = alpha_t * focal * ce_loss
        else:
            focal_loss = focal * ce_loss

        # --- reduction ------------------------------------------------------
        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        else:                      # 'none'
            return focal_loss