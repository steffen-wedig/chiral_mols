import umap
from jaxtyping import Float
from torch import Tensor
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

# Speed up rendering of large point clouds
mpl.rcParams["agg.path.chunksize"] = 10000


import numpy as np
import umap
import matplotlib.pyplot as plt


class UMAPPlotter:
    def __init__(self, embeddings, labels, sample_size_class0=None, random_state=None):
        """
        Initialize UMAPPlotter by computing UMAP components once.

        Args:
            embeddings: array-like of shape (N, EmbDim), or PyTorch Tensor.
            labels: array-like of shape (N,) with integer labels.
            sample_size_class0: int or None. Number of class 0 samples to use.
            random_state: int or RandomState for reproducibility.
        """
        # Convert to numpy arrays
        self.emb_arr = (
            embeddings.numpy()
            if hasattr(embeddings, "numpy")
            else np.asarray(embeddings)
        )
        self.labels = labels.numpy() if hasattr(labels, "numpy") else np.asarray(labels)
        self.sample_size_class0 = sample_size_class0
        self.random_state = random_state

        # Determine which indices to embed
        self.embed_idx = self._get_subsampled_indices()
        if self.random_state is not None:
            rng = np.random.RandomState(self.random_state)
            rng.shuffle(self.embed_idx)
        else:
            np.random.shuffle(self.embed_idx)

        # Subset embeddings
        self.emb_sel = self.emb_arr[self.embed_idx]

        # Compute UMAP only once
        self.umap_model = umap.UMAP()
        self.components = self.umap_model.fit_transform(
            self.emb_sel, force_all_finite=True
        )

    def _get_subsampled_indices(self):
        """
        Subsample class 0 if requested, always include classes 1 & 2.
        """
        idx0 = np.where(self.labels == 0)[0]
        idx1_2 = np.where((self.labels == 1) | (self.labels == 2))[0]
        if self.sample_size_class0 is not None and self.sample_size_class0 < len(idx0):
            sampled0 = np.random.choice(idx0, self.sample_size_class0, replace=False)
        else:
            sampled0 = idx0
        return np.concatenate([sampled0, idx1_2])

    def plot(
        self,
        color,
        color_name,
        class_names,
        rasterize=True,
        s=5,
        alpha=1,
        sample=None,
        random_state=None,
    ):
        """
        Plot the precomputed UMAP with a given color mapping, optionally with random subsampling.

        Args:
            color: array-like of length N with values to color points.
            color_name: str for legend title.
            class_names: dict mapping class integer to display name.
            rasterize: bool to rasterize for performance.
            s: marker size.
            alpha: float for transparency.
            sample: None | float in (0,1] | int >= 1
                - None: use all points
                - float: fraction of points to plot
                - int: maximum number of points to plot
            random_state: int or None
                Seed for reproducible sampling.

        Returns:
            matplotlib.figure.Figure
        """

        # Extract colors for selected indices
        color_arr = color.numpy() if hasattr(color, "numpy") else np.asarray(color)
        color_sel = color_arr[self.embed_idx]
        print(color_name)
        print(np.unique(color_sel))
        # Components already correspond to embed_idx
        comps = self.components
        n = comps.shape[0]

        # --- Subsample indices if requested ---
        if sample is not None and n > 0:
            rng = np.random.default_rng(random_state)
            if isinstance(sample, float):
                if not (0 < sample <= 1):
                    raise ValueError("When 'sample' is a float it must be in (0, 1].")
                k = max(1, int(np.ceil(n * sample)))
            else:  # treat as int
                k = min(int(sample), n)
            idx = rng.choice(n, size=k, replace=False)
            comps = comps[idx]
            color_sel = color_sel[idx]

        fig, ax = plt.subplots()
        sc = ax.scatter(
            comps[:, 0],
            comps[:, 1],
            c=color_sel,
            marker=".",
            edgecolors="none",  # absolutely no edge
            linewidths=0,  # zero line width
            s=s,
            cmap="tab20",
            alpha=alpha,
            rasterized=rasterize,
        )

        # Build legend handles with labels from the *plotted* classes
        unique_vals = np.unique(color_sel)
        handles = []
        vmax = unique_vals.max() if unique_vals.size and unique_vals.max() > 0 else 1
        cmap_obj = plt.cm.get_cmap("tab20")
        for val in unique_vals:
            label = class_names.get(int(val), str(val))
            handles.append(
                plt.Line2D(
                    [],
                    [],
                    marker="o",
                    linestyle="",
                    markersize=6,
                    markerfacecolor=cmap_obj(val / vmax),
                    markeredgecolor="none",
                    label=label,
                )
            )

        ax.legend(handles=handles, title=color_name, loc="upper right")
        ax.set_title(f"UMAP colored by {color_name}")
        ax.set_xlabel("UMAP 1")
        ax.set_ylabel("UMAP 2")
        return fig


def get_l2_histogram(chiral_embeddings, class_labels):

    hist = plt.figure()
    labels = {0: "Achiral", 1: "R", 2: "S", 3: "Error"}

    for i, c in enumerate(np.unique(class_labels)):
        emb_c = chiral_embeddings[class_labels == c]
        l2_norms = torch.linalg.norm(emb_c, dim=1).numpy()
        plt.hist(l2_norms, density=True, label=labels[c])

    plt.title(f"L2 Norm of chiral embeddings")
    plt.xlabel("L2Norm")
    plt.ylabel("Relative Frequency")
    plt.legend()

    return hist


def get_std_bar_chart(std_per_channel, channel):

    x = np.arange(len(std_per_channel))
    bar_chart = plt.figure()
    plt.bar(x, height=std_per_channel)

    plt.title(f"Std per channel Class {channel}")
    plt.xlabel("Chiral Embedding Channel (by class)")

    return bar_chart


def get_boxplot(chiral_embeddings, class_labels):

    n_dim = chiral_embeddings.shape[-1]
    fig = plt.figure(figsize=(10, 5))
    plt.title("Boxplot of channelwise distributions of chiral embeddings")
    off = np.linspace(-0.3, 0.3, len(np.unique(class_labels)))
    for i, c in enumerate(np.unique(class_labels)):
        plt.vlines(x=np.arange(1.5, 24.5, 1), ymin=-5, ymax=5, colors="k")
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
    plt.ylim([-5, 5])
    tick_labels = np.arange(1, n_dim + 1)
    plt.xticks(ticks=tick_labels, labels=tick_labels, rotation=90)

    return fig


import torch


def compute_channel_stats(chiral_embeddings, class_labels):

    figs = {}

    for c in [0, 1, 2]:

        class_embeddings = chiral_embeddings[class_labels == c]

        std_per_channel = torch.std(class_embeddings, dim=0).numpy()
        figs[f"std_bars_per_channel_{c}"] = get_std_bar_chart(std_per_channel, c)

    figs[f"l2_norm_hist"] = get_l2_histogram(chiral_embeddings, class_labels)
    figs[f"boxplot"] = get_boxplot(chiral_embeddings, class_labels)

    return figs


def get_misclassified_structure_ids(class_labels, logits, atomwise_structure_ids):

    # ensure numpy arrays
    class_labels = np.asarray(class_labels)
    logits = np.asarray(logits)
    atomwise_structure_ids = np.asarray(atomwise_structure_ids)

    # model predictions
    preds = np.argmax(logits, axis=1)

    # mask: anything non-zero (chiral) predicted as zero (achiral)
    mis_chiral_as_achiral_mask = (preds == 0) & (class_labels != 0)

    mis_achircal_as_chiral_mask = (preds != 0) & (class_labels == 0)

    # mask: R↔S swaps
    confused_r_s_mask = ((preds == 1) & (class_labels == 2)) | (
        (preds == 2) & (class_labels == 1)
    )

    # pull out the IDs
    mis_chiral_as_achiral_ids = atomwise_structure_ids[mis_chiral_as_achiral_mask]

    mis_achircal_as_chiral_ids = atomwise_structure_ids[mis_achircal_as_chiral_mask]
    confused_r_s_ids = atomwise_structure_ids[confused_r_s_mask]

    # return as lists
    return (
        mis_chiral_as_achiral_ids.tolist(),
        mis_achircal_as_chiral_ids.tolist(),
        confused_r_s_ids.tolist(),
    )
