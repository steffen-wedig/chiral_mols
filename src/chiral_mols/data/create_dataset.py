from rdkit import Chem
from rdkit.Chem import AllChem
from ase import Atoms
from typing import List, Tuple
from chiral_mols.data.structure_id import StructureID
from collections import defaultdict
from typing import List, Tuple
from itertools import chain


def chirality_codes(mol: Chem.Mol) -> list[int]:
    """
    Returns a list of length N_atoms where
      0 = not chiral,
      1 = R center,
      2 = S center.
    """
    # force‐assign all CIP tags
    Chem.AssignAtomChiralTagsFromStructure(mol)
    Chem.AssignStereochemistry(mol, cleanIt=True, force=True)

    # map each atom to 0/1/2
    return [
        (
            0
            if not atom.HasProp("_CIPCode")
            else (1 if atom.GetProp("_CIPCode") == "R" else 2)
        )
        for atom in mol.GetAtoms()
    ]


def process_single_mol(mol: Chem.Mol):
    """
    Generate ASE Atoms for each conformer, and return a list of
    identical per-atom chirality code lists (one copy per conformer).
    """
    atom_chirality = chirality_codes(mol)
    symbols = [atom.GetSymbol() for atom in mol.GetAtoms()]

    # one ASE Atoms object per conformer
    atoms = [
        Atoms(symbols=symbols, positions=conf.GetPositions(), pbc=False, info = {"smiles" : Chem.MolToSmiles(mol,allHsExplicit=False)})
        for conf in mol.GetConformers()
    ]

    # repeat (and shallow‑copy) the chirality list for each conformer
    atom_chirality_all = [list(atom_chirality) for _ in mol.GetConformers()]

    return atoms, atom_chirality_all


def get_enantionmer_id(atom_chirality : List[List[int]]) -> int:

    c = set(chain.from_iterable(atom_chirality))
    if 1 in c and 2 in c:
        raise ValueError("Both R/S in mol")
    elif 1 in c:
        return 1
    elif 2 in c: 
        return 2
    else:
        raise ValueError("No Stereocenter in mol")


def get_structures_from_mol_id(
    structure_ids: List[StructureID],
    mol_ids: List[int]
) -> List[int]:
    """
    Given a list of MoleculeIDs, return the indices in `structure_ids`
    whose .MoleculeID is in that list.
    """
    return [
        idx
        for idx, s in enumerate(structure_ids)
        if s.MoleculeID in mol_ids
    ]


def select_mols_with_complete_enantiomer(
    structure_ids: List[StructureID]
) -> List[int]:
    """
    Return all MoleculeIDs for which both enantiomer 1 and 2 appear.
    """
    # Map each MoleculeID to the set of EnantiomerIDs seen
    enant_map: dict[int, set[int]] = {}
    for s in structure_ids:
        enant_map.setdefault(s.MoleculeID, set()).add(s.EnantiomerID)

    # Keep those molecules where both 1 and 2 are in the set
    complete = [mol_id for mol_id, enants in enant_map.items()
                if {1, 2}.issubset(enants)]
    return complete


def convert_mols_to_dataset(all_mols: List[Chem.Mol]) -> Tuple[List[StructureID], List[Atoms], List[List[int]] ]:

    non_isomeric_smiles_set = {}

    all_atoms = []
    all_chirality_labels = []
    all_structure_ids = []

    running_structure_counter = 0

    for mol in all_mols:

        atoms, atom_chirality = process_single_mol(mol)
        # Get a molecule ID, lookup the nonisomeric mol id
        non_isomeric_smi = Chem.MolToSmiles(mol, isomericSmiles=False)
        if non_isomeric_smi in non_isomeric_smiles_set:
            molecule_id = non_isomeric_smiles_set[non_isomeric_smi]
        else:
            molecule_id = len(non_isomeric_smiles_set.keys())
            non_isomeric_smiles_set[non_isomeric_smi] = molecule_id


        try:
            enantiomer_id = get_enantionmer_id(atom_chirality)
        except ValueError:
            continue

        structure_ids = [
            StructureID(
                StructureID=running_structure_counter + conformer_id,
                MoleculeID=molecule_id,
                EnantiomerID=enantiomer_id,
                ConformerID=conformer_id,
            )
            for conformer_id in range(len(atoms))
        ]

        all_atoms.extend(atoms)
        all_chirality_labels.extend(atom_chirality)
        all_structure_ids.extend(structure_ids)
        running_structure_counter += len(atoms)

    assert len(set(non_isomeric_smiles_set.values())) == len(
        list(non_isomeric_smiles_set.values())
    )  # Asserts mol id uniqueness


    all_structure_ids, all_atoms, all_chirality_labels = prune_single_mols(all_structure_ids, all_atoms, all_chirality_labels)


    assert len(all_atoms) == len(all_structure_ids)
    assert len(all_atoms) == len(all_chirality_labels)

    return all_structure_ids, all_atoms, all_chirality_labels



def prune_single_mols(all_structure_ids, all_atoms, all_chirality_labels):


    # ------------------------------------------------------------------
    # Find MoleculeIDs that have *both* enantiomers
    # ------------------------------------------------------------------
    mol_to_enants = defaultdict(set)
    for sid in all_structure_ids:
        mol_to_enants[sid.MoleculeID].add(sid.EnantiomerID)

    valid_mol_ids = {mid for mid, en_set in mol_to_enants.items() if en_set == {1, 2}}

    # ------------------------------------------------------------------
    # Prune everything else
    # ------------------------------------------------------------------
    keep_idx = [
        idx for idx, sid in enumerate(all_structure_ids) if sid.MoleculeID in valid_mol_ids
    ]

    # Filter the parallel lists
    all_structure_ids = [all_structure_ids[i] for i in keep_idx]
    all_atoms          = [all_atoms[i]          for i in keep_idx]
    all_chirality_labels = [all_chirality_labels[i] for i in keep_idx]

    # (optional) re‑number StructureID.StructureID so they’re consecutive again
    for new_id, sid in enumerate(all_structure_ids):
        all_structure_ids[new_id] = StructureID(
            StructureID=new_id,
            MoleculeID=sid.MoleculeID,
            EnantiomerID=sid.EnantiomerID,
            ConformerID=sid.ConformerID,
        )

    # ------------------------------------------------------------------
    # Integrity checks
    # ------------------------------------------------------------------
    assert len(all_atoms) == len(all_structure_ids) == len(all_chirality_labels)

    return all_structure_ids, all_atoms, all_chirality_labels



def get_ase_atoms(smiles) -> Atoms:
    mol = Chem.MolFromSmiles(smiles)
    mol = Chem.AddHs(mol)
    AllChem.EmbedMolecule(
        mol, useBasicKnowledge=True, useExpTorsionAnglePrefs=True, randomSeed=-1
    )

    symbols = [atom.GetSymbol() for atom in mol.GetAtoms()]

    # one ASE Atoms object per conformer
    atoms = Atoms(symbols=symbols, positions=mol.GetConformer().GetPositions(), pbc=False, info = {"smiles" : Chem.MolToSmiles(mol,allHsExplicit=False)})
    
    

    return atoms