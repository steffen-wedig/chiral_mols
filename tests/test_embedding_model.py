import torch
import pytest
from e3nn import o3
from e3nn.o3 import Irreps

from chiral_mols.model.chiral_embedding_model import ChiralEmbeddingModel, ChiGate

torch.manual_seed(0)


def make_model(
    pseudoscalar_dim: int = 8,
    chiral_dim: int = 16,
    dtype=torch.float64,
):
    # simple irreps containing invariants (0e) and equivariants (1o)
    # invariants: 4 x 0e  -> dim = 4
    # equivariants: 2 x 1o -> dim = 2 * (2*1 + 1) = 6
    # total F = 10
    input_irreps = o3.Irreps("4x0e + 2x1o")

    model = ChiralEmbeddingModel(
        input_irreps=input_irreps,
        pseudoscalar_dimension=pseudoscalar_dim,
        chiral_embedding_dim=chiral_dim,
        mean_inv_atomic_embedding=None,
        std_inv_atomic_embedding=None,
        gated=True,
        equivariant_rms_norm=True,
        dtype=dtype,
    ).to(dtype)

    return model, input_irreps


def random_atomic_embeddings(B, N, irreps: Irreps, dtype):
    F = irreps.dim
    if B is None:  # unbatched
        return torch.randn(N, F, dtype=dtype)
    return torch.randn(B, N, F, dtype=dtype)


def test_forward_unbatched_shape_and_dtype():
    model, irreps = make_model()
    N = 5
    x = random_atomic_embeddings(None, N, irreps, model.dtype)

    out = model(x)  # (N, C)

    assert out.shape == (N, model.linear_out.out_features)
    assert out.dtype == torch.float32  # coerced at the end


def test_forward_batched_shape_and_padding_zeroing():
    model, irreps = make_model()
    B, N = 3, 7
    x = random_atomic_embeddings(B, N, irreps, model.dtype)

    # mark last 2 atoms in each molecule as padded
    padding = torch.zeros(B, N, dtype=torch.bool)
    padding[:, -2:] = True

    out = model(x, padding=padding)  # (B, N, C)
    C = model.linear_out.out_features

    assert out.shape == (B, N, C)
    # padded atoms should be exactly zero
    assert torch.all(out[:, -2:, :].abs() < 1e-12)


def test_batched_vs_unbatched_equivalence_without_padding():
    model, irreps = make_model()
    B, N = 1, 6
    x_batched = random_atomic_embeddings(B, N, irreps, model.dtype)
    x_unbatched = x_batched.squeeze(0)

    out_batched = model(x_batched)       # (1, N, C)
    out_unbatched = model(x_unbatched)   # (N, C)

    assert torch.allclose(out_batched.squeeze(0), out_unbatched, atol=1e-6, rtol=1e-6)


def test_backward_pass_grads_flow():
    model, irreps = make_model()
    B, N = 2, 4
    x = random_atomic_embeddings(B, N, irreps, model.dtype)
    padding = torch.zeros(B, N, dtype=torch.bool)

    out = model(x, padding=padding)  # (B, N, C)
    loss = out.pow(2).mean()
    loss.backward()

    # ensure every parameter that requires grad received a gradient
    for name, p in model.named_parameters():
        if p.requires_grad:
            assert p.grad is not None, f"{name} has no grad"


class ZeroGate(torch.nn.Module):
    """Used to ensure gating truly gates the signal."""
    def __init__(self, K):
        super().__init__()
        self.K = K
    def forward(self, inv):
        return torch.zeros(inv.shape[0], self.K, device=inv.device, dtype=inv.dtype)


def test_zero_gate_zeroes_output():
    model, irreps = make_model()
    B, N = 2, 3
    x = random_atomic_embeddings(B, N, irreps, model.dtype)

    # replace gate with a zero gate => output must be all zeros
    K = model.pseudoscalar_irreps.dim
    model.chi_gate = ZeroGate(K)

    out = model(x)  # (B, N, C)
    assert torch.allclose(out, torch.zeros_like(out), atol=1e-12)