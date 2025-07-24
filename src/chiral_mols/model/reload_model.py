import pydantic_yaml as pydyaml
import torch 
from chiral_mols.model.chiral_embedding_model import ChiralEmbeddingModel
from chiral_mols.model.configuration import ChiralEmbeddingConfig
from pathlib import Path

def reload_model_from_dir(model_dir : Path):
    chiral_embedding_model_config = pydyaml.parse_yaml_file_as(ChiralEmbeddingConfig, file = model_dir / "chiral_embedding_model_config.yaml")


    chiral_embedding_model = ChiralEmbeddingModel(
            **chiral_embedding_model_config.model_dump(exclude="reload_state_dict"),
            mean_inv_atomic_embedding=None,
            std_inv_atomic_embedding=None,
            dtype=torch.float64,
        )
    state_dict = torch.load(chiral_embedding_model_config.reload_state_dict)
    chiral_embedding_model.load_state_dict(state_dict)  
    return chiral_embedding_model