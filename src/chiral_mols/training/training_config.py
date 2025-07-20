from pydantic import BaseModel, Field



class TrainConfig(BaseModel):
    batch_size: int = Field(512, description="Number of samples per batch")
    learning_rate: float = Field(1e-3, description="Step size for the optimizer")
    N_epochs: int = Field(100, description="Total number of training epochs")
 