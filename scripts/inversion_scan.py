from __future__ import annotations
from e3nn.o3 import Irreps
import argparse
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.EnumerateStereoisomers import (
    EnumerateStereoisomers,
    StereoEnumerationOptions,
)
import torch
from ase import Atoms
from ase.io import Trajectory, write
from mace.calculators import MACECalculator
from chiral_mols.model.chiral_embedding_model import ChiralEmbeddingModel
from chiral_mols.model.molecular_embedding_model import MolecularEmbeddingModel


@dataclass
class StereoPair:
    label: str  # "R" or "S"
    mol: Chem.Mol
    atoms: Atoms  # ASE representation


# -----------------------------------------------------------------------------
# Basic helpers (unchanged from previous revision)
# -----------------------------------------------------------------------------

def smiles_to_optimised_rdkit_mol(smiles: str, seed: int = 2025) -> Chem.Mol:
    mol = Chem.MolFromSmiles(smiles, sanitize=True)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")
    mol = Chem.AddHs(mol)
    if AllChem.EmbedMolecule(mol, randomSeed=seed, clearConfs=True):
        raise RuntimeError("RDKit 3-D embedding failed. Try another seed or molecule.")
    try:
        AllChem.MMFFOptimizeMolecule(mol)
    except Exception:
        pass
    return mol


def rdkit_mol_to_ase_atoms(mol: Chem.Mol) -> Atoms:
    conf = mol.GetConformer()
    pos = np.array([list(conf.GetAtomPosition(i)) for i in range(mol.GetNumAtoms())])
    syms = [a.GetSymbol() for a in mol.GetAtoms()]
    return Atoms(symbols=syms, positions=pos)


def find_first_chiral_centre(mol: Chem.Mol) -> int:
    centres = Chem.FindMolChiralCenters(
        mol, includeUnassigned=True, useLegacyImplementation=False
    )
    if not centres:
        raise ValueError("No chiral centres detected – cannot build rotation path.")
    return centres[0][0]


def assign_cip(mol: Chem.Mol) -> None:
    Chem.AssignStereochemistry(mol, cleanIt=True, force=True)


def enumerate_r_s(smiles: str) -> Dict[str, Chem.Mol]:
    base = Chem.MolFromSmiles(smiles, sanitize=True)
    if base is None:
        raise ValueError("Invalid SMILES input")

    opts = StereoEnumerationOptions(unique=True, onlyUnassigned=True)
    isomers = list(EnumerateStereoisomers(base, options=opts)) or [base]
    labelled: Dict[str, Chem.Mol] = {}
    for iso in isomers:
        assign_cip(iso)
        for idx, cip in Chem.FindMolChiralCenters(
            iso, includeUnassigned=False, useLegacyImplementation=False
        ):
            if cip in {"R", "S"} and cip not in labelled:
                labelled[cip] = iso
        if len(labelled) == 2:
            break

    # Fallback: flip first centre if we still have only one.
    if len(labelled) == 1:
        single_cip, mol_single = next(iter(labelled.items()))
        idx = find_first_chiral_centre(mol_single)
        mol_flipped = Chem.RWMol(mol_single)
        atom = mol_flipped.GetAtomWithIdx(idx)
        tag = atom.GetChiralTag()
        flip_map = {
            Chem.CHI_TETRAHEDRAL_CW: Chem.CHI_TETRAHEDRAL_CCW,
            Chem.CHI_TETRAHEDRAL_CCW: Chem.CHI_TETRAHEDRAL_CW,
        }
        if tag not in flip_map:
            raise RuntimeError("Unexpected chiral tag – cannot flip stereo.")
        atom.SetChiralTag(flip_map[tag])
        mol_flipped = mol_flipped.GetMol()
        assign_cip(mol_flipped)
        labelled["S" if single_cip == "R" else "R"] = mol_flipped

    if set(labelled) != {"R", "S"}:
        raise RuntimeError("Failed to obtain both enantiomers – molecule may not be chiral?")
    return labelled


def make_stereo_pair(smiles: str) -> Dict[str, StereoPair]:
    stereos = enumerate_r_s(smiles)
    out: Dict[str, StereoPair] = {}
    for lab, iso in stereos.items():
        iso3d = smiles_to_optimised_rdkit_mol(Chem.MolToSmiles(iso))
        out[lab] = StereoPair(label=lab, mol=iso3d, atoms=rdkit_mol_to_ase_atoms(iso3d))
    return out


def list_neighbours(mol: Chem.Mol, centre: int) -> List[int]:
    return [n.GetIdx() for n in mol.GetAtomWithIdx(centre).GetNeighbors()]

# -----------------------------------------------------------------------------
# Rotation helpers
# -----------------------------------------------------------------------------

def _rotation_matrix(axis: np.ndarray, angle: float) -> np.ndarray:
    """Return 3×3 Rodrigues rotation matrix for *unit* axis and angle (rad)."""
    axis = axis / np.linalg.norm(axis)
    kx, ky, kz = axis
    c = np.cos(angle)
    s = np.sin(angle)
    v = 1 - c
    return np.array(
        [
            [kx * kx * v + c, kx * ky * v - kz * s, kx * kz * v + ky * s],
            [ky * kx * v + kz * s, ky * ky * v + c, ky * kz * v - kx * s],
            [kz * kx * v - ky * s, kz * ky * v + kx * s, kz * kz * v + c],
        ]
    )


def build_rotation_trajectory(
    atoms: Atoms,
    centre_idx: int,
    pair_indices: Tuple[int, int],
    n_frames: int = 21,
) -> List[Atoms]:
    """Rotate the two chosen neighbours 0→π around the bisector axis.

    * atoms: ASE Atoms of starting (R) geometry
    * centre_idx: index of chiral centre atom
    * pair_indices: tuple of the two neighbour atom indices to be rotated
    * n_frames: number of snapshots including endpoints
    """
    a_idx, b_idx = pair_indices
    if a_idx == b_idx:
        raise ValueError("pair_indices must be two different neighbour atoms")

    centre = atoms.positions[centre_idx]
    vec_a = atoms.positions[a_idx] - centre
    vec_b = atoms.positions[b_idx] - centre

    axis = vec_a + vec_b  # bisector
    norm = np.linalg.norm(axis)
    if norm < 1e-6:
        raise ValueError(
            "Selected neighbours appear colinear/antiparallel; bisector axis ill‑defined."
        )
    axis /= norm

    frames = []
    for t, ang in enumerate(np.linspace(0.0, np.pi, n_frames)):
        R = _rotation_matrix(axis, ang)
        new_positions = atoms.positions.copy()
        for idx, vec in ((a_idx, vec_a), (b_idx, vec_b)):
            new_positions[idx] = centre + R @ vec
        frames.append(Atoms(symbols=atoms.get_chemical_symbols(), positions=new_positions))
    return frames


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate a rotational inversion trajectory (R→S) with ASE."
    )
    parser.add_argument("smiles", help="Input SMILES string")
    parser.add_argument(
        "--frames",
        type=int,
        default=21,
        help="Number of frames in trajectory (default 21)",
    )
    parser.add_argument(
        "--outfile",
        default="rotation.traj",
        help="Output ASE trajectory filename",
    )
    parser.add_argument(
        "--rotate",
        nargs=2,
        type=int,
        metavar=("IDX_A", "IDX_B"),
        help="Specify two neighbour atom indices to rotate (default: first two)",
    )
    args = parser.parse_args()

    pair = make_stereo_pair(args.smiles)
    r_atoms = pair["R"].atoms
    centre_idx = find_first_chiral_centre(pair["R"].mol)
    neighbours = list_neighbours(pair["R"].mol, centre_idx)

    print(f"Chiral centre: atom {centre_idx} ({pair['R'].mol.GetAtomWithIdx(centre_idx).GetSymbol()})")
    for n in neighbours:
        print(f"  – neighbour {n}: {pair['R'].mol.GetAtomWithIdx(n).GetSymbol()}")

    if args.rotate is None:
        pair_indices = tuple(neighbours[:2])  # naive default
        print(f"Using default neighbour pair {pair_indices} for rotation.")
    else:
        pair_indices = tuple(args.rotate)
        if not all(i in neighbours for i in pair_indices):
            raise ValueError("--rotate indices must be neighbour atoms of the chiral centre")

    frames = build_rotation_trajectory(
        atoms=r_atoms,
        centre_idx=centre_idx,
        pair_indices=pair_indices,  # type: ignore
        n_frames=args.frames,
    )

    write("trajectory.xyz", frames)
    

    chiral_embedding_model = ChiralEmbeddingModel(
    input_irreps=Irreps("128x0e+128x1o+128x0e"),
    pseudoscalar_irreps=Irreps("128x0o"),
    output_embedding_dim=128,
    mean_inv_atomic_embedding=None,
    std_inv_atomic_embedding=None,
    low_dim_equivariant=128
)
    
    MACE_PATH = "/share/snw30/projects/mace_model/MACE-OFF24_medium.model"
    mace_calc = MACECalculator(MACE_PATH, device= "cuda", default_dtype= "float32")
    model_filename = "/share/snw30/projects/chiral_mols/models/chiral_embedding_model.pth"
    chiral_embedding_model.load_state_dict(torch.load(model_filename))
    
    me_model = MolecularEmbeddingModel(mace_calc=mace_calc, chiral_embeding_model= chiral_embedding_model)


    from ase.build import molecule
    atoms = molecule("H2COH")

    print(atoms.get_positions())

    mirrored_atoms = atoms.copy()
    mirrored_atoms.set_positions(-1 * atoms.get_positions())

    print(chiral_embedding_model(mirrored_atoms) - chiral_embedding_model(atoms))




    chiral_embeddings = []
    from ase.io import read
    frames = read("/share/snw30/projects/chiral_mols/dataset/afloqualone.xyz", ":")

    print(len(frames))
    for atoms in frames: 
        chiral_embeddings.append(me_model(atoms)[2, :])

    chiral_embeddings = torch.stack(chiral_embeddings, dim = 0).detach().cpu().numpy()
    print(chiral_embeddings.shape)
    

    import matplotlib.pyplot as plt

    x = np.arange(chiral_embeddings.shape[0])
    fig = plt.figure()
    plt.plot(x, chiral_embeddings[:,])



    fig.savefig("ps_plot.png")


    
    

    




if __name__ == "__main__":
    main()
