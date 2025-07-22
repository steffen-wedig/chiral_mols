import warnings
import os

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
from chiral_mols.model.chiral_embedding_model import ChiralEmbeddingModel
from chiral_mols.model.classifier import ChiralityClassifier
from torch.optim import AdamW
from chiral_mols.training.evaluation import build_metric_collection, wandb_confmat


from chiral_mols.training.training import (
    train_one_epoch,
    evaluate,
    parse_args,
    make_training_dir,
)
from chiral_mols.training.loss import get_class_weights, FocalLoss
from torch.optim.lr_scheduler import OneCycleLR
from chiral_mols.model.configuration import (
    ChiralEmbeddingConfig,
    ChiralityClassifierConfig,
)
import wandb
import pydantic_yaml as pydyaml

run_name = parse_args()
torch.manual_seed(0)
training_data_dir = make_training_dir(run_name)


dataset_dir = Path("/share/snw30/projects/chiral_mols/dataset/chiral_atoms")
training_cfg = TrainConfig(batch_size=256, learning_rate=8e-4, N_epochs=50)
N_classes = 3


chiral_embedding_model_config = ChiralEmbeddingConfig(
    input_irreps="128x0e+128x1o+128x0e",
    pseudoscalar_dimension=64,
    chiral_embedding_dim=24,
    gated=True,
    equivariant_rms_norm=True,
)

classifier_config = ChiralityClassifierConfig(
    chiral_embedding_dim=chiral_embedding_model_config.chiral_embedding_dim,
    hidden_dim=256,
    n_classes=N_classes,
    dropout=0.3,
)

print("Start loading the dataset")
dataset = PtrMoleculeDataset.reload_dataset_from_dir(dataset_dir)
print("Loaded the dataset")
dataset_splitting = DatasetSplitter(dataset.structure_ids)
train_idx, val_idx = dataset_splitting.random_split_by_molecule(
    train_val_ratios=[0.7, 0.3]
)
train_data = Subset(dataset, train_idx)
val_data = Subset(dataset, val_idx)

train_data_loader = DataLoader(
    train_data,
    batch_size=training_cfg.batch_size,
    collate_fn=concat_collate,
    shuffle=True,
    drop_last=True,
)
val_data_loader = DataLoader(
    val_data, batch_size=training_cfg.batch_size, collate_fn=concat_collate
)

mean_inv, std_inv = get_mean_std_invariant_indices(
    train_data.dataset.embeddings, chiral_embedding_model_config.input_irreps
)

device = "cuda"

chiral_embedding_model = ChiralEmbeddingModel(
    **chiral_embedding_model_config.model_dump(exclude_none= True),
    mean_inv_atomic_embedding=mean_inv,
    std_inv_atomic_embedding=std_inv,
    dtype=torch.float32,
)


classifier = ChiralityClassifier(**classifier_config.model_dump(exclude_none=True))


weights = get_class_weights(train_data.dataset, num_classes=3, scheme="balanced")
loss_fn = FocalLoss(gamma=2.0, alpha=weights, reduction="mean").to(device)


optimizer = AdamW(
    params=list(classifier.parameters()) + list(chiral_embedding_model.parameters()),
    lr=training_cfg.learning_rate,
)


# 3. Instantiate the OneCycleLR scheduler
scheduler = OneCycleLR(
    optimizer,
    max_lr=training_cfg.learning_rate,  # peak learning rate
    total_steps=training_cfg.N_epochs * len(train_data_loader),
)


cfg = {
    "training_config": training_cfg,
    "classifier_config": classifier_config.model_dump_json(),
    "chiral_embedding_model_config": chiral_embedding_model_config.model_dump_json(),
}

wandb.init(
    project="chirality_prediction",
    entity="threedscriptors",
    name=run_name,
    config=cfg,
)


classifier.to(device)
chiral_embedding_model.to(device)

# Write configs ready for reloading to disk:


classifier_config.reload_state_dict = Path(f"{training_data_dir}/classifier.pth")
pydyaml.to_yaml_file(f"{training_data_dir}/classifier_config.yaml", classifier_config)

chiral_embedding_model_config.reload_state_dict = Path(
    f"{training_data_dir}/chiral_embedding_model.pth"
)
pydyaml.to_yaml_file(
    f"{training_data_dir}/chiral_embedding_model_config.yaml",
    chiral_embedding_model_config,
)


torch.save({
    "state_dict": loss_fn.state_dict(), 
    "gamma" : loss_fn.gamma,  # buffers: gamma, alpha, ...
    "reduction": loss_fn.reduction,       # plain attribute, not in state_dict
}, f"{training_data_dir}/focal_loss_state.pth")




best_val_loss = torch.inf
metrics = build_metric_collection(N_classes).to(device=device)

for epoch in range(1, training_cfg.N_epochs + 1):

    train_loss = train_one_epoch(
        chiral_embedding_model,
        classifier,
        train_data_loader,
        loss_fn,
        optimizer,
        scheduler,
        device,
    )

    val_stats, _ = evaluate(
        chiral_embedding_model,
        classifier,
        val_data_loader,
        metrics,
        loss_fn,
        device,
    )

    print(f"\nEpoch {epoch}")
    print(f"  train_loss         : {train_loss:.4f}")
    for k, v in val_stats.items():
        if k != "confmat":
            print(f"  {k:18s}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")
    print("  confusion matrix:\n", val_stats["confmat"].numpy())

    wandb.log(
        {
            "epoch": epoch,
            "train/loss": train_loss,
            **{f"val/{k}": v for k, v in val_stats.items() if k != "confmat"},
            "val/confmat": wandb_confmat(val_stats["confmat"]),
        }
    )

    if val_stats["cross_entropy"] < best_val_loss:
        best_val_loss = val_stats["cross_entropy"]

        torch.save(
            chiral_embedding_model.state_dict(),
            f"{training_data_dir}/chiral_embedding_model.pth",
        )

        torch.save(classifier.state_dict(), f"{training_data_dir}/classifier.pth")
        print(f" Epoch {epoch}: New best model saved.")




