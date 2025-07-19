import pytest
from chiral_mols.data.structure_id import StructureID
from chiral_mols.training.dataset_splitting import DatasetSplitter

@pytest.fixture
def dummy_structure_ids():
    """
    Create 6 dummy molecules (IDs 0-5), each with 2 conformers,
    so total 12 StructureID objects.
    """
    ids = []
    for mol_id in range(6):
        for conf in range(2):
            ids.append(StructureID(
                StructureID=mol_id*10 + conf,
                MoleculeID=mol_id,
                EnantiomerID=0,
                ConformerID=conf
            ))
    return ids

def test_no_overlap_between_splits(dummy_structure_ids):
    splitter = DatasetSplitter(dummy_structure_ids)
    # Use shuffle=False for determinism
    splits = splitter.random_split_by_molecule((0.5, 0.5), shuffle=False)

    # Build set of MoleculeIDs in each split
    mol_sets = [
        {dummy_structure_ids[i].MoleculeID for i in split}
        for split in splits
    ]

    # Check pairwise disjointness
    for i in range(len(mol_sets)):
        for j in range(i+1, len(mol_sets)):
            inter = mol_sets[i].intersection(mol_sets[j])
            assert not inter, (
                f"Splits {i} and {j} both contain MoleculeID(s): {inter}"
            )