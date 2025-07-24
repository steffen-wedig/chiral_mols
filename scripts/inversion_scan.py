from chiral_mols.model.configuration import ChiralEmbeddingConfig
from chiral_mols.model.chiral_embedding_model import ChiralEmbeddingModel
from mace.calculators import MACECalculator
import torch
import pydantic_yaml as pydyaml
from chiral_mols.model.molecular_embedding_model import MolecularEmbeddingModel
from rdkit import Chem
from rdkit.Chem.rdMolTransforms import GetDihedralDeg, SetDihedralDeg
import numpy as np
from rdkit.Chem import AllChem, Draw
from ase import Atoms
from rdkit.Geometry import Point3D
from pathlib import Path

from rdkit.Chem import rdchem
smiles = "CC(C)Cc1ccc(cc1)[C@@H](C)C(=O)O"

mol = Chem.MolFromSmiles(smiles)
mol = Chem.AddHs(mol)

centre = 10 

def mmff_energy(mol, conf_id):
    mp = AllChem.MMFFGetMoleculeProperties(mol, mmffVariant="MMFF94s")
    ff = AllChem.MMFFGetMoleculeForceField(mol, mp, confId=conf_id)
    return ff.CalcEnergy()



def enumerate_sidechains(mol, core_idx_set):
    """
    Given an RDKit mol and a set of core atom indices,
    returns a list of side‑chains, where each side‑chain
    is a list of atom indices not in core_idx_set but
    connected to it.
    """
    visited = set()
    sidechains = []
    for core_idx in core_idx_set:
        core_atom = mol.GetAtomWithIdx(core_idx)
        for nbr in core_atom.GetNeighbors():
            nbr_idx = nbr.GetIdx()
            if nbr_idx in core_idx_set or nbr_idx in visited:
                continue
            # BFS from this neighbor
            chain = []
            stack = [nbr]
            while stack:
                atom = stack.pop()
                ai = atom.GetIdx()
                if ai in core_idx_set or ai in visited:
                    continue
                visited.add(ai)
                chain.append(ai)
                for nn in atom.GetNeighbors():
                    if nn.GetIdx() not in core_idx_set and nn.GetIdx() not in visited:
                        stack.append(nn)
            sidechains.append(chain)
    return sidechains


sidechains = sorted(enumerate_sidechains(mol, [10]), key = lambda x : len(x))


frozen_atoms = sidechains[0] +sidechains[2]

chain_1 = sidechains[1]

chain_2 = sidechains[2]

AllChem.EmbedMolecule(mol)

conf0 = mol.GetConformer()

# --- build axis ---
p0 = conf0.GetAtomPosition(centre)
p1 = conf0.GetAtomPosition(chain_1[0])
p2 = conf0.GetAtomPosition(chain_2[0])

r_c1 = np.array([p1.x - p0.x, p1.y - p0.y, p1.z - p0.z])
r_c2 = np.array([p2.x - p0.x, p2.y - p0.y, p2.z - p0.z])

r_c1 /= np.linalg.norm(r_c1)
r_c2 /= np.linalg.norm(r_c2)

axis = r_c1 + r_c2          # bisector axis (pick whatever makes chemical sense)
axis /= np.linalg.norm(axis)
origin = np.array([p0.x, p0.y, p0.z])

def axis_angle_matrix(axis, theta):
    """Rodrigues rotation formula -> 3x3 rotation matrix"""
    axis = axis / np.linalg.norm(axis)
    ux, uy, uz = axis
    c = np.cos(theta)
    s = np.sin(theta)
    R = np.array([[c+ux*ux*(1-c),     ux*uy*(1-c)-uz*s, ux*uz*(1-c)+uy*s],
                  [uy*ux*(1-c)+uz*s,  c+uy*uy*(1-c),    uy*uz*(1-c)-ux*s],
                  [uz*ux*(1-c)-uy*s,  uz*uy*(1-c)+ux*s, c+uz*uz*(1-c)]])
    return R

def rotate_subset(conf, atom_ids, origin, axis, angle_deg):
    R = axis_angle_matrix(axis, np.deg2rad(angle_deg))
    for i in atom_ids:
        p = conf.GetAtomPosition(i)
        v = np.array([p.x, p.y, p.z]) - origin
        v = R @ v + origin
        conf.SetAtomPosition(i, Point3D(*v))

angles = np.arange(0, 181, 3)
print(f"Angle Steps : {len(angles)}")
subset = chain_1 + chain_2
energies = []
base_coords = np.array(conf0.GetPositions())

for ang in angles:
    # new conformer from base coords
    new_conf = Chem.Conformer(mol.GetNumAtoms())
    for i, (x,y,z) in enumerate(base_coords):
        new_conf.SetAtomPosition(i, Point3D(x,y,z))

    cid = mol.AddConformer(new_conf, assignId=True)
    Chem.AssignStereochemistryFrom3D(mol, confId=cid )
    print(Chem.FindMolChiralCenters(mol, includeUnassigned=True, useLegacyImplementation=False))

    rotate_subset(mol.GetConformer(cid), subset, origin, axis, ang)
    e = mmff_energy(mol, cid)
    energies.append(e)

# now energies[i] corresponds to angles[i] for conformer i added last
for a,e in zip(angles, energies):
    print(f"{a:3.0f}°  {e:10.3f} kcal/mol")


ase_frames = []

atom_symbols = [a.GetSymbol() for a in mol.GetAtoms()]



for conf in mol.GetConformers():
    ase_frames.append(Atoms(atom_symbols, positions=conf.GetPositions(), pbc=False))


from ase.io import write

write("rotation.xyz", ase_frames)

model_dir = Path("/share/snw30/projects/chiral_mols/training_runs/6-2025_07_22_20_52_17-NoLinear")
chiral_embedding_model_config = pydyaml.parse_yaml_file_as(ChiralEmbeddingConfig, file = model_dir / "chiral_embedding_model_config.yaml")


chiral_embedding_model = ChiralEmbeddingModel(
    **chiral_embedding_model_config.model_dump(exclude="reload_state_dict"),
    mean_inv_atomic_embedding=None,
    std_inv_atomic_embedding=None,
    dtype=torch.float32,
)
state_dict = torch.load(chiral_embedding_model_config.reload_state_dict)
#chiral_embedding_model.load_state_dict(state_dict)

MACE_PATH = "/share/snw30/projects/mace_model/MACE-OFF24_medium.model"
mace_calc = MACECalculator(model_paths=MACE_PATH, enable_cueq=True, device="cuda", default_dtype="float32")

model = MolecularEmbeddingModel(mace_calc=mace_calc, chiral_embeding_model=chiral_embedding_model)

stereocenter_embeddings = []
for atoms in ase_frames:
    chiral_embeddings = model(atoms)
    stereocenter_embeddings.append(chiral_embeddings[centre, :])

stereocenter_embeddings = torch.stack(stereocenter_embeddings).detach().cpu().numpy()



import matplotlib.pyplot as plt


fig = plt.figure()
plt.plot(stereocenter_embeddings[:,:])

fig.savefig("pseudoscalars.png")