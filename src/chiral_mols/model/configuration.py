from pydantic import BaseModel, ConfigDict, Field
from e3nn.o3 import Irreps
from threedscriptors.configuration.architecture_config import IrrepType
from pathlib import Path


class ChiralEmbeddingConfig(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    input_irreps: IrrepType = Irreps("128x0e+128x1o+128x0e")
    pseudoscalar_dimension: int = 128  #
    chiral_embedding_dim: int = 128  # Linear Projection from Pseudosclars to dim.
    gated: bool = True
    equivariant_rms_norm: bool = True
    reload_state_dict: Path | None = None


class ChiralityClassifierConfig(BaseModel):
    chiral_embedding_dim: int = Field(
        32, gt=0, description="Size of the input chiral embedding dimension"
    )
    hidden_dim: int = Field(
        64, gt=0, description="Size of the hidden layers in the MLP"
    )
    n_classes: int = Field(3, gt=0, description="Number of output classes")
    dropout: float = Field(
        0.1, ge=0.0, le=1.0, description="Dropout probability between 0 and 1"
    )
    reload_state_dict: Path | None = None
