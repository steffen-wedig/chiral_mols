import torch
from torchmetrics import MetricCollection

from torchmetrics.classification import (
    Accuracy,  # overall + balanced (macro) accuracy
    Precision,
    Recall,
    F1Score,  # per‑class & macro scores
    ConfusionMatrix,  # full confusion matrix
    CohenKappa,  # Cohen's κ
    MatthewsCorrCoef,  # MCC
    AUROC,  # one‑vs‑rest ROC AUC
    AveragePrecision,  # one‑vs‑rest PR AUC
)
import matplotlib.pyplot as plt
import wandb
from torchmetrics.classification import BinaryRecall




class ChiralRecall(BinaryRecall):
    """
    Recall over *chiral* centres only (labels 1 and 2).
    """

    def __init__(self, threshold: float = 0.5):
        super().__init__(threshold=threshold)      # standard binary recall

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        # 1.  Ground‑truth: 1 = chiral (R or S), 0 = achiral
        target_chiral = (target > 0).long()        # *** keep it integer ***

        # 2.  Model score: P(is chiral) = P(R) + P(S)
        prob_chiral = preds[:, 1:].sum(dim=1)      # shape (B,)

        # 3.  Let BinaryRecall do the book‑keeping
        return super().update(prob_chiral, target_chiral)




def build_metric_collection(num_classes):


    metrics = MetricCollection(
        {
            "accuracy": Accuracy(task="multiclass", num_classes=num_classes),
            "macro_f1": F1Score(
                task="multiclass", num_classes=num_classes, average="macro"
            ),
            "mcc": MatthewsCorrCoef(task="multiclass", num_classes=num_classes),
            "cohen_kappa": CohenKappa(task="multiclass", num_classes=num_classes),
            "auroc_macro": AUROC(
                task="multiclass", num_classes=num_classes, average="macro"
            ),
            "auprc_macro": AveragePrecision(
                task="multiclass", num_classes=num_classes, average="macro"
            ),
            "confmat": ConfusionMatrix(task="multiclass", num_classes=num_classes),
            "chiral_recall" : ChiralRecall(),
        }
    )

    return metrics



def wandb_confmat(cm_tensor, class_names=("Achiral", "R", "S")):
    cm = cm_tensor.cpu().numpy()

    fig, ax = plt.subplots(figsize=(4, 4))
    im = ax.imshow(cm, interpolation="nearest")
    ax.set_xticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticks(range(len(class_names)))
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")

    # add counts
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, f"{cm[i, j]}", ha="center", va="center", color="w")

    ax.set_title("Confusion matrix")
    fig.tight_layout()

    img = wandb.Image(fig)
    plt.close(fig)
    return img