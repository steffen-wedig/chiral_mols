from chiral_mols.data.read_cmrt_data import extract_smiles, write_smiles_list_to_file, read_smiles_from_file
from pathlib import Path


if __name__ == "__main__":
    # From a file on disk:
    smiles_list = extract_smiles(Path("dataset/raw_data/cmrt_raw_data.csv"))
    
    filename = "/share/snw30/projects/chiral_mols/dataset/raw_data/cmrt_smiles"

    write_smiles_list_to_file(smiles_list, filename)

    new_smiles_list = read_smiles_from_file(filename)
    assert smiles_list == new_smiles_list