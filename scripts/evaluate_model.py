import warnings
from ase.data import chemical_symbols
from ase.visualize.plot import plot_atoms

warnings.filterwarnings(
    "ignore",
    message=r"The TorchScript type system doesn't support instance-level annotations.*",
    category=UserWarning,
)
import torch
from chiral_mols.data.ptr_dataset import PtrMoleculeDataset
from chiral_mols.evaluation.smiles_error_analysis import (
    analyse_smiles_list_cip_distance,
)
import numpy as np
from torch.utils.data import DataLoader
from chiral_mols.data.sample import concat_collate

from pathlib import Path
from e3nn.o3 import Irreps
from chiral_mols.model.chiral_embedding_model import ChiralEmbeddingModel
from chiral_mols.model.classifier import ChiralityClassifier

from chiral_mols.training.evaluation import build_metric_collection
from chiral_mols.training.training import evaluate
from chiral_mols.training.loss import FocalLoss
from chiral_mols.model.configuration import (
    ChiralEmbeddingConfig,
    ChiralityClassifierConfig,
)
import pydantic_yaml as pydyaml
from chiral_mols.evaluation.evaluation import UMAPPlotter, compute_channel_stats
import yaml

import matplotlib.pyplot as plt
from chiral_mols.evaluation.evaluation import get_misclassified_structure_ids

torch.manual_seed(0)


dataset_dir = Path("/share/snw30/projects/chiral_mols/dataset/chiral_atoms_120k")
N_classes = 3

print("Reloading Dataset")
dataset = PtrMoleculeDataset.reload_dataset_from_dir(dataset_dir, reload_atoms=True)
print("Loaded the dataset")

dataloader = DataLoader(
    dataset, batch_size=1024, collate_fn=concat_collate, shuffle=False, drop_last=False
)
model_dir = Path(
    "/share/snw30/projects/chiral_mols/training_runs/13-2025_07_23_15_08_22-OutLinear"
)

device = "cuda"

classifier_config = pydyaml.parse_yaml_file_as(
    ChiralityClassifierConfig, file=model_dir / "classifier_config.yaml"
)
chiral_embedding_model_config = pydyaml.parse_yaml_file_as(
    ChiralEmbeddingConfig, file=model_dir / "chiral_embedding_model_config.yaml"
)


chiral_embedding_model = ChiralEmbeddingModel(
    **chiral_embedding_model_config.model_dump(exclude="reload_state_dict"),
    mean_inv_atomic_embedding=None,
    std_inv_atomic_embedding=None,
    dtype=torch.float32,
)

classifier = ChiralityClassifier(
    **classifier_config.model_dump(exclude="reload_state_dict")
)
classifier.load_state_dict(torch.load(classifier_config.reload_state_dict))


state_dict = torch.load(chiral_embedding_model_config.reload_state_dict)
chiral_embedding_model.load_state_dict(state_dict)


loss_fn_params = torch.load(model_dir / "focal_loss_state.pth", map_location=device)

loss_fn = FocalLoss(  # create shell with any values that
    gamma=loss_fn_params["gamma"],
    alpha=None,  # will be overwritten by load_state_dict
    reduction=loss_fn_params["reduction"],
)
loss_fn.load_state_dict(loss_fn_params["state_dict"], strict=False)
loss_fn.to(device)


classifier.to(device)
chiral_embedding_model.to(device)


metrics = build_metric_collection(N_classes).to(device=device)


val_stats, model_output = evaluate(
    chiral_embedding_model,
    classifier,
    dataloader,
    metrics,
    loss_fn,
    device,
    return_full_output=True,
)

val_stats["confmat"] = val_stats["confmat"].numpy()
for k, v in val_stats.items():
    if k != "confmat":
        print(f"  {k:18s}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")
print("  confusion matrix:\n", val_stats["confmat"])
with open(f"{model_dir}/test_stats.yaml", "w") as f:
    yaml.dump(val_stats, f)


#umap_plotter = UMAPPlotter(
#    model_output.chiral_embedding, model_output.class_labels, sample_size_class0=0
#)
#
#
#reference_fig = umap_plotter.plot(
#    color=model_output.class_labels,
#    color_name="reference_labels",
#    class_names={0: "Achiral", 1: "R", 2: "S"},
#)
#reference_fig.savefig(f"{model_dir}/reference_umap.png", dpi=400)
#plt.close(reference_fig)
#
#class_fig = umap_plotter.plot(
#    color=model_output.predictions,
#    color_name="Model Predictions",
#    class_names={0: "Achiral", 1: "R", 2: "S"},
#)
#class_fig.savefig(f"{model_dir}/predicted_class_umap.png", dpi=400)
#plt.close(class_fig)
#
#missclassified_labels = model_output.class_labels != model_output.predictions
#error_fig = umap_plotter.plot(
#    color=missclassified_labels,
#    color_name="Model Predictions",
#    class_names={0: "Correct Class", 1: "Incorrect Class"},
#)
#error_fig.savefig(f"{model_dir}/error_umap.png", dpi=400)
#
#
#
#print(dataset.atomic_numbers.cpu().tolist()[:200])
#
## get unique atomic numbers as ints
#unique_atom_types = set(dataset.atomic_numbers.cpu().tolist())
#
#from ase.data import colors
## build a dict mapping Python ints → symbols
#atom_type_dict = {z: colors.cpk_colors[z] for z in unique_atom_types}
#
#
#atom_types = [colors.cpk_colors[i] for i in dataset.atomic_numbers]
#
#atom_type_fig = umap_plotter.plot(
#    color=atom_types, color_name="Atom Type", class_names=atom_type_dict, sample = 50000
#)
#
#atom_type_fig.savefig(f"{model_dir}/atom_type_umap.png", dpi=400)
#
#
figs = compute_channel_stats(model_output.chiral_embedding, model_output.class_labels)


for figname, fig in figs.items():
    fig.savefig(f"{model_dir}/{figname}.png")


mis_chiral_as_achiral_ids, mis_achiral_as_chiral_ids, confused_r_s_ids = (
    get_misclassified_structure_ids(
        class_labels=model_output.class_labels,
        logits=model_output.model_logits,
        atomwise_structure_ids=dataset.atomwise_structure_ids,
    )
)

atom_type_counts = np.bincount(dataset.atomic_numbers[mis_achiral_as_chiral_ids])

print(atom_type_counts)

fig = plt.figure()
plt.bar(np.arange(len(atom_type_counts)), atom_type_counts)
plt.xlabel("Element of misclassified achiral as chiral")
fig.savefig(f"{model_dir}/misclassified_true_achirals_by_element.png")


confused_r_s_smiles = [dataset.smiles[i] for i in confused_r_s_ids]
confused_true_chiral_as_achiral_smiles = [dataset.smiles[i] for i in mis_chiral_as_achiral_ids]


confused_r_s_atoms = [dataset.get_atoms_from_structure_id(i) for i in list(set(confused_r_s_ids))]

print(len(confused_r_s_atoms))


def plot_many_atoms(atoms):
        
    N_horizontal = 3
    N_vertical = (len(atoms) // 3 )+1
    fig, axarr = plt.subplots(N_vertical, N_horizontal)
     
    fig.set_figheight(4*N_vertical)
    fig.set_figwidth(4*N_horizontal)
    for i, mol in enumerate(atoms):
        plot_atoms(mol, axarr[i // 3 , i % 3])
    return fig


fig = plot_many_atoms(confused_r_s_atoms[:200])

fig.savefig(f"{model_dir}/relaxed_atoms.png")




cip_hop_distances = analyse_smiles_list_cip_distance(confused_true_chiral_as_achiral_smiles)


distance_counts = np.bincount(cip_hop_distances)


fig = plt.figure()
plt.bar(np.arange(len(distance_counts)), distance_counts)
plt.xlabel("Min Hop distance that determines CIP assignment")
fig.savefig(f"{model_dir}/cip_distance_counts.png")
