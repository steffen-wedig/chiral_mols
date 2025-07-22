import warnings


warnings.filterwarnings(
    "ignore",
    message=r"The TorchScript type system doesn't support instance-level annotations.*",
    category=UserWarning,
)
import torch
from chiral_mols.data.ptr_dataset import PtrMoleculeDataset
from chiral_mols.training.dataset_splitting import DatasetSplitter
from torch.utils.data import DataLoader, Subset
from chiral_mols.training.training_config import TrainConfig
from chiral_mols.data.sample import ptr_collate_padding, concat_collate
from chiral_mols.training.embedding_normalization import get_mean_std_invariant_indices
from pathlib import Path
from e3nn.o3 import Irreps
from chiral_mols.model.chiral_embedding_model import ChiralEmbeddingModel
from chiral_mols.model.classifier import ChiralityClassifier
from torch.optim import AdamW
from torch.nn import CrossEntropyLoss
from chiral_mols.training.evaluation import build_metric_collection, wandb_confmat
from chiral_mols.training.training import train_one_epoch, evaluate, parse_args
from chiral_mols.training.loss import get_class_weights, FocalLoss
from torch.optim.lr_scheduler import OneCycleLR
from chiral_mols.model.configuration import ChiralEmbeddingConfig, ChiralityClassifierConfig
import pydantic_yaml as pydyaml
import wandb


torch.manual_seed(0)


dataset_dir = Path("/share/snw30/projects/chiral_mols/dataset/chiral_atoms")
N_classes = 3

dataset = PtrMoleculeDataset.reload_dataset_from_dir(dataset_dir)
print("Loaded the dataset")

dataloader = DataLoader(
    dataset,
    batch_size=1024,
    collate_fn=concat_collate,
)
model_dir = Path("/share/snw30/projects/chiral_mols/training_runs/3-2025_07_22_18_54_40-CEDim=24")

device = "cuda"

classifier_config = pydyaml.parse_yaml_file_as(ChiralityClassifierConfig, file = model_dir / "classifier_config.yaml")
chiral_embedding_model_config = pydyaml.parse_yaml_file_as(ChiralEmbeddingConfig, file = model_dir / "chiral_embedding_model_config.yaml")


chiral_embedding_model = ChiralEmbeddingModel(
    **chiral_embedding_model_config.model_dump(exclude="reload_state_dict"),
    mean_inv_atomic_embedding=None,
    std_inv_atomic_embedding=None,
    dtype=torch.float32,
)

classifier = ChiralityClassifier(**classifier_config.model_dump(exclude = "reload_state_dict"))
classifier.load_state_dict(torch.load(classifier_config.reload_state_dict))


state_dict = torch.load(chiral_embedding_model_config.reload_state_dict)
chiral_embedding_model.load_state_dict(state_dict)


loss_fn_params = torch.load(model_dir / "focal_loss_state.pth", map_location=device)

loss_fn = FocalLoss(                      # create shell with any values that
    gamma=loss_fn_params["gamma"], alpha=None,                # will be overwritten by load_state_dict
    reduction=loss_fn_params["reduction"]
)
loss_fn.load_state_dict(loss_fn_params["state_dict"], strict= False)
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
    )

for k, v in val_stats.items():
    if k != "confmat":
        print(f"  {k:18s}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")
print("  confusion matrix:\n", val_stats["confmat"].numpy())
    
from chiral_mols.evaluation.evaluation import plot_umap_components, compute_channel_stats


#reference_fig, model_pred_fig = plot_umap_components(model_output.chiral_embedding, model_output.class_labels, model_output.model_logits, sample_size_class0=0)
#
#reference_fig.savefig("no_achiral_reference_umap.png")
#model_pred_fig.savefig("no_achiral_model_umap.png")
#

figs = compute_channel_stats(model_output.chiral_embedding, model_output.class_labels)


for figname, fig in figs.items():
    fig.savefig(f"{figname}.png")