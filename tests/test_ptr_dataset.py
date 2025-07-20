from pathlib import Path

import numpy as np
import pytest
import torch
from chiral_mols.data.structure_id import StructureID
from chiral_mols.data.ptr_dataset import PtrMoleculeDataset
from torch.utils.data import DataLoader
from chiral_mols.data.sample import ptr_collate_padding, concat_collate
from typing import Tuple


class _RandomDataFactory:
    def __init__(self, rng_seed: int = 0):
        self.rng = np.random.default_rng(rng_seed)

    def make(self, num_mols: int = 5, min_atoms: int = 1, max_atoms: int = 8, F: int = 4) -> Tuple[PtrMoleculeDataset,np.ndarray]:
        ids = [
            StructureID(
                StructureID=i,
                MoleculeID=i,
                EnantiomerID=0,
                ConformerID=0,
            )
            for i in range(num_mols)
        ]
        lengths = self.rng.integers(low=min_atoms, high=max_atoms + 1, size=num_mols)
        emb_list = [self.rng.normal(size=(n, F)).astype(np.float32) for n in lengths]
        lbl_list = [self.rng.integers(0, 2, size=n).astype(np.int64) for n in lengths]
        return PtrMoleculeDataset.from_dataset_list(ids, emb_list, lbl_list), lengths



@pytest.fixture(scope="module")
def random_dataset():
    factory = _RandomDataFactory(rng_seed=42)
    return factory.make()


def test_round_trip(tmp_path: Path, random_dataset):
    dataset, _ = random_dataset
    dataset.store_dataset(tmp_path)
    ds2 = PtrMoleculeDataset.reload_dataset_from_dir(tmp_path)
    assert torch.allclose(dataset.embeddings, ds2.embeddings)
    assert torch.equal(dataset.labels, ds2.labels)
    assert torch.equal(dataset.ptr, ds2.ptr)
    assert dataset.structure_ids == ds2.structure_ids


def test_collate_padding(random_dataset):
    dataset, lengths = random_dataset
    loader = DataLoader(dataset, batch_size=len(dataset), collate_fn=ptr_collate_padding, shuffle=False)
    batch = next(iter(loader))
    B = len(dataset)
    max_n = max(lengths)

    # Shape checks
    assert batch.embeddings.shape == (B, max_n, dataset.embeddings.shape[1])
    assert batch.chirality_labels.shape == (B, max_n)
    assert batch.atom_mask.shape == (B, max_n)

    # Mask sums equal original lengths
    assert torch.equal(batch.atom_mask.sum(dim=1), torch.tensor(lengths))

    # Padded regions are zero
    pad_regions = ~batch.atom_mask
    assert torch.all(batch.embeddings[pad_regions] == 0)
    assert torch.all(batch.chirality_labels[pad_regions] == 0)


def test_sample_to_(random_dataset):
    dataset, _ = random_dataset
    sample = dataset[0]
    cpu = torch.device("cpu")
    sample.to_(device=cpu)
    assert sample.embeddings.device == cpu
    assert sample.chirality_labels.device == cpu
    if sample.atom_mask is not None:
        assert sample.atom_mask.device == cpu




def test_concat_collate_shapes_and_content(random_dataset):
    """
    Given a list of `Sample`s (one per molecule) concat_collate should:
        • concatenate atoms along dim=0
        • keep feature dim (F) unchanged
        • produce a 1-D chirality label tensor
        • produce a mol_idx tensor whose unique values are 0…B-1
          and appear in contiguous blocks (no interleaving)
    """
    dataset, lengths = random_dataset                       # fixture output
    molecules = [dataset[i] for i in range(len(dataset))]   # list[Sample]

    flat = concat_collate(molecules)                        # Forward

    # ---------- basic shape checks ----------
    n_total = int(np.sum(lengths))
    F = molecules[0].embeddings.shape[1]

    assert flat.embeddings.shape == (n_total, F)
    assert flat.chirality_labels.shape == (n_total,)
    assert flat.mol_idx.shape == (n_total,)

    # ---------- mol_idx should cover 0 … B‑1 ----------
    unique_ids = flat.mol_idx.unique(sorted=True)
    assert torch.equal(unique_ids, torch.arange(len(lengths)))

    # ---------- mol_idx should be piecewise‑constant blocks ----------
    # (diff == 0 inside a molecule, 1 at the boundary ➜ exactly B‑1 jumps)
    jumps = (flat.mol_idx[1:] - flat.mol_idx[:-1]).nonzero(as_tuple=False)
    assert jumps.numel() == len(lengths) - 1

    # ---------- content equality ----------
    cursor = 0
    for mol_id, sample in enumerate(molecules):
        n = sample.embeddings.shape[0]

        # embeddings and labels match exactly
        assert torch.allclose(
            flat.embeddings[cursor : cursor + n], sample.embeddings
        )
        assert torch.equal(
            flat.chirality_labels[cursor : cursor + n], sample.chirality_labels
        )

        # mol_idx correctly filled
        assert torch.equal(
            flat.mol_idx[cursor : cursor + n],
            torch.full((n,), mol_id, dtype=flat.mol_idx.dtype),
        )
        cursor += n