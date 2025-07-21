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

import wandb


torch.manual_seed(0)


dataset_dir = Path("/share/snw30/projects/chiral_mols/dataset/chiral_atoms")


N_classes = 3


input_irreps = Irreps("128x0e+128x1o+128x0e")


dataset = PtrMoleculeDataset.reload_dataset_from_dir(dataset_dir)
print("Loaded the dataset")



datloader = DataLoader(
    dataset,
    batch_size=1024,
    collate_fn=concat_collate,
)


device = "cuda"

chiral_embedding_model = ChiralEmbeddingModel(
    input_irreps=input_irreps,
    pseudoscalar_irreps=Irreps("64x0o"),
    output_embedding_dim=chiral_embedding_dim,
    mean_inv_atomic_embedding=mean_inv,
    std_inv_atomic_embedding=std_inv,
    low_dim_equivariant=64
)

classifier = ChiralityClassifier(
    chiral_embedding_dim=chiral_embedding_dim,
    hidden_dim=256,
    n_classes=N_classes,
    dropout=0.3,
)



total_steps = training_cfg.N_epochs * len(train_data_loader)

# 3. Instantiate the OneCycleLR scheduler
scheduler = OneCycleLR(
    optimizer,
    max_lr=training_cfg.learning_rate,   # peak learning rate
    total_steps=total_steps,             # total number of optimizer steps
    pct_start=0.3,                       # fraction of cycle spent increasing LR
    anneal_strategy='cos',              # cosine annealing (other option: 'linear')
    div_factor=25.0,                     # initial lr = max_lr/div_factor
    final_div_factor=1e4,                # final lr = initial_lr/final_div_factor
    three_phase=False,                   # set True for 3-phase schedule
    last_epoch=-1,                       # leave as default unless resuming
)


classifier.to(device)
chiral_embedding_model.to(device)


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

    val_stats = evaluate(
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