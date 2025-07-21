import polars as pl
from typing import List, Union
from pathlib import Path
from typing import Iterator

def extract_smiles(csv_source: Union[str, Path]) -> List[str]:
    """
    Read a CSV (file path or in-memory buffer) and return the SMILES column as a list.
    """
    # Read CSV into a Polars DataFrame
    df = pl.read_csv(csv_source)
    # Select the SMILES column and convert to a Python list
    return df.select("SMILES").to_series().to_list()

def write_smiles_list_to_file(smiles_list : List[str], filename : Path | str):

    with open(filename, "w") as f:
        for smi in smiles_list:
            f.write(smi + "\n")


def read_smiles_from_file(filename : Path | str):

    with open(filename, "r") as f:
        smiles_list = f.readlines()

        smiles_list = [smi.split()[0] for smi in smiles_list]
    return smiles_list



def batch_smiles(smiles_list: List[str], batch_size: int) -> Iterator[List[str]]:
    """
    Yield successive batches of SMILES strings from a list.

    :param smiles_list: List of SMILES strings.
    :param batch_size: Number of SMILES per batch.
    :return: Iterator over lists of SMILES strings (each of length ≤ batch_size).
    """
    for i in range(0, len(smiles_list), batch_size):
        yield smiles_list[i : i + batch_size]