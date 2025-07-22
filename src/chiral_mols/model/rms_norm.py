from torch import nn
import torch



class RMSLayerNorm(nn.Module):
    def __init__(self, num_irreps: int, eps: float = 1e-6):
        """
        RMS-style layer norm for equivariant (type-L) blocks.

        Args:
          num_irreps: number of irreducible blocks C
          eps: small constant to avoid div/0
        """
        super().__init__()
        # one learnable scale per channel/block
        self.gamma = nn.Parameter(torch.ones(num_irreps, 1))
        self.eps = eps

    def forward(self, S_e: torch.Tensor) -> torch.Tensor:
        """
        Args:
          x: tensor of shape (..., C, D)

        Returns:
          normalized tensor of the same shape
        """
        # 1) compute per-block L2 norm over the last dim:
        #    shape: (..., C, 1)

        B, D = S_e.shape
        C = D // 3

        S_e = S_e.view(B, C, 3)

        block_norms = torch.linalg.norm(S_e, dim=-1, keepdim=True)

        # 2) compute RMS of those norms *across* the C irreps:
        #    shape: (..., 1, 1)
        rms = torch.sqrt(
            torch.mean(block_norms.pow(2), dim=-2, keepdim=True) + self.eps
        )

        # 3) divide each block by the shared rms and apply per-channel scale
        #    broadcasting gamma over any leading dims and over D
        out = (S_e / rms) * self.gamma


        return out.reshape(B, D)