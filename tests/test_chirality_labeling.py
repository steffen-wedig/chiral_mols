from rdkit import Chem
from chiral_mols.create_dataset import chirality_codes


def test_achiral_molecule():
    # Ethane has no stereocenters
    mol = Chem.MolFromSmiles("CC")
    assert chirality_codes(mol) == [0, 0]


def test_lactic_acid_s_configuration():
    # L-(+)-Lactic acid: central carbon is S
    mol = Chem.MolFromSmiles("C[C@H](O)C(=O)O")
    codes = chirality_codes(mol)
    # Only atom index 1 is chiral and S (2)
    assert codes[1] == 2
    # All other atoms are non-chiral
    assert sum(codes) == 2


def test_lactic_acid_r_configuration():
    # D-(-)-Lactic acid: central carbon is R
    mol = Chem.MolFromSmiles("C[C@@H](O)C(=O)O")
    codes = chirality_codes(mol)
    # Only atom index 1 is chiral and R (1)
    assert codes[1] == 1
    # All other atoms are non-chiral
    assert sum(codes) == 1
