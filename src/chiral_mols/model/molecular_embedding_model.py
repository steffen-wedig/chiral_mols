import torch 

from mace.calculators import MACECalculator
from chiral_mols.model.chiral_embedding_model import ChiralEmbeddingModel
from ase import Atoms
from chiral_mols.model.irreps_utils import get_mace_calculator_irrep_signature

class MolecularEmbeddingModel(torch.nn.Module):


    def __init__(self, mace_calc : MACECalculator, chiral_embeding_model : ChiralEmbeddingModel):


        super().__init__()

        self.mace_calc = mace_calc
        self.chiral_embedding_model = chiral_embeding_model

        assert get_mace_calculator_irrep_signature(self.mace_calc) == self.chiral_embedding_model.input_irreps

    def forward(self, ase_atoms: Atoms) :
        

        descriptors = torch.from_numpy(self.mace_calc.get_descriptors(ase_atoms, invariants_only= False))

        
        descriptors = descriptors.to(dtype=self.chiral_embedding_model.dtype)
        pseudoscalar_embeddings = self.chiral_embedding_model(descriptors)

        return pseudoscalar_embeddings