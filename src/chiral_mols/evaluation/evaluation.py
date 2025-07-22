import umap
from jaxtyping import Float
from torch import Tensor
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

# Speed up rendering of large point clouds
mpl.rcParams['agg.path.chunksize'] = 10000


def get_umap_components(
    embeddings: Float[Tensor, "N UmapDim"]
) -> np.ndarray:
    """
    Compute UMAP projections for the input embeddings.

    Args:
        embeddings: Tensor or array of shape (N, EmbDim).

    Returns:
        A NumPy array of shape (N, 2) with UMAP components.
    """
    umap_calc = umap.UMAP()
    umap_components = umap_calc.fit_transform(
        embeddings, force_all_finite=True
    )
    return umap_components


def plot_umap_components(
    chiral_embeddings: Float[Tensor, "N UmapDim"],
    class_labels,
    logits,
    sample_size_class0: int = None,
    rasterize: bool = True
):
    """
    Generate two UMAP scatter plots where all points from classes 1 & 2 are shown,
    and optionally only a random subset of class 0 is embedded and plotted.

    Args:
        chiral_embeddings: Tensor or array of shape (N, EmbDim).
        class_labels: Array-like of length N with true integer labels.
        logits: Tensor or array of shape (N, C), raw model outputs.
        sample_size_class0: Number of points to randomly sample from class 0 before UMAP.
                            If None or >= count of class 0, use all class 0 points.
        rasterize: Whether to rasterize the scatter layers for faster rendering.

    Returns:
        Tuple of (reference_fig, prediction_fig).
    """
    # Convert labels and logits to numpy arrays
    if hasattr(class_labels, 'detach'):
        labels = class_labels.detach().cpu().numpy()
    else:
        labels = np.asarray(class_labels)
    if hasattr(logits, 'argmax'):
        preds = logits.argmax(dim=1).detach().cpu().numpy()
    else:
        preds = np.argmax(np.asarray(logits), axis=1)

    # Identify indices for each class
    idx_class0 = np.where(labels == 0)[0]
    idx_class1_2 = np.where((labels == 1) | (labels == 2))[0]

    # Sample from class 0 indices if requested
    if sample_size_class0 is not None and sample_size_class0 < len(idx_class0):
        sampled0 = np.random.choice(idx_class0, sample_size_class0, replace=False)
    else:
        sampled0 = idx_class0

    # Combine indices for embedding
    embed_idx = np.concatenate([sampled0, idx_class1_2])
    # Shuffle to avoid ordering artifacts
    np.random.shuffle(embed_idx)

    # Subset the embeddings, labels, and preds
    # Handle torch Tensor input for embeddings
    if hasattr(chiral_embeddings, 'detach'):
        emb_arr = chiral_embeddings.detach().cpu().numpy()
    else:
        emb_arr = np.asarray(chiral_embeddings)
    emb_sel = emb_arr[embed_idx]
    labels_sel = labels[embed_idx]
    preds_sel = preds[embed_idx]

    # Compute UMAP on subset
    comps_sel = get_umap_components(emb_sel)

    # Plot True Labels
    reference_fig, ax1 = plt.subplots()
    sc1 = ax1.scatter(
        comps_sel[:, 0], comps_sel[:, 1],
        c=labels_sel,
        cmap='tab10',
        s=1,
        alpha=0.8,
        edgecolors='none',
        rasterized=rasterize
    )
    ax1.legend(*sc1.legend_elements(), title="True Classes", loc='upper right', markerscale=6)
    ax1.set_title("UMAP: True Labels (all class 1 & 2, sampled class 0 before embedding)")
    ax1.set_xlabel("UMAP 1")
    ax1.set_ylabel("UMAP 2")

    # Plot Predicted Labels
    prediction_fig, ax2 = plt.subplots()
    sc2 = ax2.scatter(
        comps_sel[:, 0], comps_sel[:, 1],
        c=preds_sel,
        cmap='tab10',
        s=1,
        alpha=0.8,
        edgecolors='none',
        rasterized=rasterize
    )
    ax2.legend(*sc2.legend_elements(), title="Predicted Classes", loc='upper right', markerscale=6)
    ax2.set_title("UMAP: Predicted Labels (all class 1 & 2, sampled class 0 before embedding)")
    ax2.set_xlabel("UMAP 1")
    ax2.set_ylabel("UMAP 2")


    return reference_fig, prediction_fig




def get_l2_histogram(chiral_embeddings, class_labels):

    hist = plt.figure()
    labels = {0 : "Achiral", 1: "R", 2: "S"}

    for i, c in enumerate(np.unique(class_labels)):
        emb_c = chiral_embeddings[class_labels == c]
        l2_norms = torch.linalg.norm(emb_c, dim= 1).numpy()
        plt.hist(l2_norms, density=True, label=labels[c])

    plt.title(f"L2 Norm of chiral embeddings")
    plt.xlabel("L2Norm")
    plt.ylabel("Relative Frequency")
    plt.legend()


    return hist

def get_std_bar_chart(std_per_channel, channel):

    x = np.arange(len(std_per_channel))
    bar_chart = plt.figure()
    plt.bar(x, height= std_per_channel)

    plt.title(f"Std per channel Class {channel}")
    plt.xlabel("Chiral Embedding Channel (by class)")

    return bar_chart

def get_boxplot(chiral_embeddings, class_labels):

    n_dim = chiral_embeddings.shape[-1]
    fig = plt.figure(figsize=(10,5))
    plt.title("Boxplot of channelwise distributions of chiral embeddings")
    off = np.linspace(-0.3, 0.3, len(np.unique(class_labels)))
    for i, c in enumerate(np.unique(class_labels)):
        plt.vlines(x = np.arange(1.5,24.5,1), ymin = -5, ymax = 5,colors="k")
        emb_c = chiral_embeddings[class_labels == c]
        # use positions shifted by 'off' to avoid overplot
        pos = np.arange(1, n_dim + 1) + off[i]
        plt.boxplot(
            emb_c,
            positions=pos,
            widths=0.2,
            showfliers=False,
            patch_artist=True,
            boxprops=dict(alpha=0.3),
        )
    plt.xlabel("Chiral Embedding Channel (by class)")
    plt.ylim([-5,5])
    tick_labels = np.arange(1,n_dim+1)
    plt.xticks(ticks = tick_labels, labels = tick_labels, rotation=90)

    return fig


import torch

def compute_channel_stats(chiral_embeddings, class_labels):


    
    figs = {}
    
    for c in [0,1,2]:
        
        class_embeddings = chiral_embeddings[class_labels == c]

        std_per_channel = torch.std(class_embeddings, dim= 0).numpy()
        figs[f"std_bars_per_channel_{c}"] = get_std_bar_chart(std_per_channel, c)
        
    figs[f"l2_norm_hist"] = get_l2_histogram(chiral_embeddings, class_labels)
    figs[f"boxplot"] = get_boxplot(chiral_embeddings, class_labels)


    return figs



























    std_channel_class_fig = plt.figure(figsize=(figsize[0], figsize[1] * 0.6))
    off = np.linspace(-0.3, 0.3, len(np.unique(class_labels)))
    for i, c in enumerate(np.unique(class_labels)):
        emb_c = emb_plot[class_labels == c]
        # use positions shifted by 'off' to avoid overplot
        pos = np.arange(1, n_dim + 1) + off[i]
        plt.boxplot(
            emb_c,
            positions=pos,
            widths=0.2,
            showfliers=False,
            patch_artist=True,
            boxprops=dict(alpha=0.3),
        )
    plt.title("Descriptor distribution per channel by class")
    plt.xlabel("Channel index")
    plt.ylabel("Descriptor value")
    plt.tight_layout()

    return total_l2, class_l2 , std_channel_fig, std_channel_class_fig
