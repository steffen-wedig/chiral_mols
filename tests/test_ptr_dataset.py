from pathlib import Path

import numpy as np
import pytest
import torch
from chiral_mols.data.structure_id import StructureID
from chiral_mols.data.ptr_dataset import PtrMoleculeDataset
from torch.utils.data import DataLoader
from chiral_mols.data.sample import ptr_collate
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
    loader = DataLoader(dataset, batch_size=len(dataset), collate_fn=ptr_collate, shuffle=False)
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