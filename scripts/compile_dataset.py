from chiral_mols.smiles_enumeration import (
    get_chiral_smiles,
    add_enantiomer_pair,
    stream_random_ligand_smiles_batches,
)
from chiral_mols.relax_mols import convert_smiles_to_mols
from tqdm import tqdm
from chiral_mols.mlip_utils import get_mace_embeddings
from mace.calculators import MACECalculator
from chiral_mols.create_dataset import convert_mols_to_dataset


MACE_PATH = "/share/snw30/projects/mace_model/MACE-OFF24_medium.model"
def main():
    tsv_file = "/share/snw30/projects/threedscriptor/3DMolecularDescriptors/data/raw_data/BindingDB_All.tsv"

    all_mols = []
    N_molecules = 100
    with tqdm(total=N_molecules) as pbar:
        for smiles in stream_random_ligand_smiles_batches(
            tsv_file, read_batch_size=500, pool_size=25000, yield_batch_size=1000
        ):

            smiles = get_chiral_smiles(smiles)
            smiles = add_enantiomer_pair(smiles)
            new_mols = convert_smiles_to_mols(smiles, N_conformers=1, verbose=False)

            pbar.update(len(new_mols))
            if new_mols is not []:
                all_mols.extend(new_mols)

            if len(all_mols) > N_molecules: 
                break


    print(all_mols)

    all_structure_ids, all_atoms, all_chirality_labels = convert_mols_to_dataset(all_mols)


    mace_calc = MACECalculator(model_paths=MACE_PATH, enable_cueq= True, device = "cuda")
    all_mace_embeddings = get_mace_embeddings(atoms=all_atoms, mace_calc=mace_calc)

    

if __name__ == "__main__":
    main()
