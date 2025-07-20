from chiral_mols.data.smiles_enumeration import (
    get_chiral_smiles,
    add_enantiomer_pair,
    stream_random_ligand_smiles_batches,
    filter_smiles,
)
from pathlib import Path
from chiral_mols.data.relax_mols import convert_smiles_to_mols
from tqdm import tqdm
from chiral_mols.data.mlip_utils import get_mace_embeddings
from mace.calculators import MACECalculator
from chiral_mols.data.create_dataset import convert_mols_to_dataset
from chiral_mols.data.ptr_dataset import PtrMoleculeDataset

MACE_PATH = "/share/snw30/projects/mace_model/MACE-OFF24_medium.model"
TSV_FILE = "/share/snw30/projects/threedscriptor/3DMolecularDescriptors/data/raw_data/BindingDB_All.tsv"
DATASET_DIR = "/share/snw30/projects/chiral_mols/dataset/chiral_atoms/"


def main():

    all_mols = []
    N_molecules = 10000

    seen_smiles = set()
    with tqdm(total=N_molecules) as pbar:
        for smiles in stream_random_ligand_smiles_batches(
            TSV_FILE, read_batch_size=500, pool_size=25000, yield_batch_size=1000
        ):

            smiles = get_chiral_smiles(smiles)
            smiles = add_enantiomer_pair(smiles)

            smiles = filter_smiles(smiles)
            smiles = [s for s in smiles if s not in seen_smiles]
            seen_smiles.update(smiles)

            new_mols = convert_smiles_to_mols(smiles, N_conformers=5, verbose=True)

            pbar.update(len(new_mols))
            if new_mols is not []:
                all_mols.extend(new_mols)

            if len(all_mols) > N_molecules:
                break

    

    all_structure_ids, all_atoms, all_chirality_labels = convert_mols_to_dataset(
        all_mols
    )


    mace_calc = MACECalculator(model_paths=MACE_PATH, enable_cueq=True, device="cuda")
    all_mace_embeddings = get_mace_embeddings(atoms=all_atoms, mace_calc=mace_calc)

    dataset = PtrMoleculeDataset.from_dataset_list(
        all_structure_ids, all_mace_embeddings, all_chirality_labels
    )

    dataset.store_dataset(Path(DATASET_DIR))


if __name__ == "__main__":
    main()
