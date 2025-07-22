from dataclasses import dataclass
import torch 


@dataclass
class EvalOutput:
    chiral_embedding: torch.Tensor
    model_logits: torch.Tensor 
    class_labels: torch.Tensor