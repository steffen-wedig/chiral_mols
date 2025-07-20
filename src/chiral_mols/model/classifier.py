import torch

from torch import nn 



class ChiralityClassifier(nn.Module):

    def __init__(
        self,
        chiral_embedding_dim: int = 32,
        hidden_dim: int = 64,
        n_classes: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        # Simple 2‐layer MLP
        self.net = nn.Sequential(
            nn.Linear(chiral_embedding_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, n_classes),
        )

    def forward(self, chiral_embedding: torch.Tensor) -> torch.Tensor:
        """
        Args:
            chiral_embedding: Tensor of shape (batch_size, n_atoms, chiral_embedding_dim)
        Returns:
            logits: Tensor of shape (batch_size, n_atoms, n_classes)
        """
        # Linear layers in PyTorch apply over the last dim, so
        # shape is preserved except for the feature dimension.
        logits = self.net(chiral_embedding)
        return logits


        