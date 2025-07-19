from chiral_mols.data.create_dataset import (
    StructureID,
    select_mols_with_complete_enantiomer,
    get_structures_from_mol_id,
)


def test_select_mols_with_complete_enantiomer_empty():
    # no data → no complete molecules
    assert select_mols_with_complete_enantiomer([]) == []


def test_select_mols_with_complete_enantiomer_incomplete():
    # Molecule 1 only has enantiomer 1; Molecule 2 only has 2
    data = [
        StructureID(StructureID=1, MoleculeID=1, EnantiomerID=1, ConformerID=0),
        StructureID(StructureID=2, MoleculeID=2, EnantiomerID=2, ConformerID=0),
    ]
    assert select_mols_with_complete_enantiomer(data) == []


def test_select_mols_with_complete_enantiomer_complete():
    # Molecule 10 has both 1 & 2; Molecule 20 only has 1
    data = [
        StructureID(StructureID=1, MoleculeID=10, EnantiomerID=1, ConformerID=0),
        StructureID(StructureID=2, MoleculeID=10, EnantiomerID=2, ConformerID=0),
        StructureID(StructureID=3, MoleculeID=20, EnantiomerID=1, ConformerID=0),
    ]
    result = select_mols_with_complete_enantiomer(data)
    assert result == [10]


def test_get_structures_from_mol_id_empty():
    # no structures → empty indices
    assert get_structures_from_mol_id([], [1, 2, 3]) == []


def test_get_structures_from_mol_id_single():
    data = [
        StructureID(StructureID=1, MoleculeID=5, EnantiomerID=1, ConformerID=0),
        StructureID(StructureID=2, MoleculeID=6, EnantiomerID=1, ConformerID=0),
    ]
    # only index 1 matches mol 6
    assert get_structures_from_mol_id(data, [6]) == [1]


def test_get_structures_from_mol_id_multiple():
    data = [
        StructureID(StructureID=1, MoleculeID=5, EnantiomerID=1, ConformerID=0),
        StructureID(StructureID=2, MoleculeID=6, EnantiomerID=1, ConformerID=0),
        StructureID(StructureID=3, MoleculeID=5, EnantiomerID=2, ConformerID=0),
        StructureID(StructureID=4, MoleculeID=7, EnantiomerID=1, ConformerID=0),
    ]
    # Looking for molecules 5 and 7 → indices 0, 2, 3
    assert get_structures_from_mol_id(data, [5, 7]) == [0, 2, 3]