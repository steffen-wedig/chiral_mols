import pytest
import torch
from torch import nn

# Import the Classifier - adjust the import path as needed
from chiral_mols.model.classifier import ChiralityClassifier


def test_classifier_output_shape_and_probs():
    """
    Test that the classifier produces logits of the expected shape
    and that softmax probabilities sum to 1 along the class dimension.
    """
    batch_size = 2
    n_atoms = 10
    embedding_dim = 32
    n_classes = 3

    # Random input tensor
    x = torch.randn(batch_size, n_atoms, embedding_dim)

    # Instantiate the model
    model = ChiralityClassifier(
        chiral_embedding_dim=embedding_dim,
        hidden_dim=64,
        n_classes=n_classes,
        dropout=0.0,
    )

    # Forward pass
    logits = model(x)

    # Check output shape
    assert logits.shape == (batch_size, n_atoms, n_classes)

    # Check that softmax probabilities sum to 1
    probs = torch.softmax(logits, dim=-1)
    sums = probs.sum(dim=-1)
    assert torch.allclose(sums, torch.ones(batch_size, n_atoms), atol=1e-6)


