import torch
from torch import nn
from torch.optim import AdamW
from torchmetrics import MetricCollection

from chiral_mols.model.chiral_embedding_model import ChiralEmbeddingModel
from chiral_mols.model.classifier import ChiralityClassifier

import argparse

def parse_args():
    """
    Parse command-line arguments and return the run_name.
    """
    parser = argparse.ArgumentParser(
        description="Parse the --run_name argument for naming runs"
    )
    parser.add_argument(
        "--run_name",
        type=str,
        required=True,
        help="Name of the run (e.g., experiment identifier)",
    )
    args = parser.parse_args()
    return args.run_name


run_name = parse_args()

def train_one_epoch(
        embedding_model: ChiralEmbeddingModel,
        classifier     : ChiralityClassifier,
        dataloader,
        loss_fn        : nn.Module,
        optimizer      : torch.optim.Optimizer,
        scheduler,
        device: str = "cuda",
):
    embedding_model.train()
    classifier.train()

    running_loss = 0.0
    n_samples    = 0

    for batch in dataloader:
        batch = batch.to_(device=device)

        optimizer.zero_grad()
        logits = classifier(embedding_model(batch.embeddings))
        loss   = loss_fn(logits, batch.chirality_labels)

        loss.backward()
        optimizer.step()
        scheduler.step()

        running_loss += loss.item() * batch.chirality_labels.numel()
        n_samples    += batch.chirality_labels.numel()

    return running_loss / n_samples






@torch.no_grad()
def evaluate(
        embedding_model: ChiralEmbeddingModel,
        classifier     : ChiralityClassifier,
        dataloader,
        metrics        : MetricCollection,
        loss_fn        : nn.Module,
        device: str = "cuda",
):
    embedding_model.eval()
    classifier.eval()
    metrics.reset()

    running_loss = 0.0
    n_samples    = 0

    for batch in dataloader:
        batch = batch.to_(device=device)
        logits = classifier(embedding_model(batch.embeddings))
        probs  = torch.softmax(logits, dim=1)

        metrics.update(probs, batch.chirality_labels)

        running_loss += loss_fn(logits, batch.chirality_labels).item() * batch.chirality_labels.numel()
        n_samples    += batch.chirality_labels.numel()

    result = metrics.compute()
    result["cross_entropy"] = running_loss / n_samples
    
    # split out the confusion matrix
    confmat = result.pop("confmat")          # tensor (C, C)

    # convert scalars to Python numbers
    result = {
        k: (v.item() if torch.is_tensor(v) and v.numel() == 1 else v)
        for k, v in result.items()
    }

    # add the confusion matrix back (as CPU tensor or numpy)
    result["confmat"] = confmat.cpu()

    return result