
import warnings


warnings.filterwarnings(
    "ignore",
    message=r"The TorchScript type system doesn't support instance-level annotations.*",
    category=UserWarning,
)
import torch
from chiral_mols.data.ptr_dataset import PtrMoleculeDataset
from chiral_mols.training.dataset_splitting import DatasetSplitter
from torch.utils.data import DataLoader, Subset
from chiral_mols.training.traininig_config import TrainConfig
from chiral_mols.data.sample import ptr_collate_padding, concat_collate
from chiral_mols.training.embedding_normalization import get_mean_std_invariant_indices
from pathlib import Path
from e3nn.o3 import Irreps
from chiral_mols.model.chiral_embedding_model import ChiralEmbeddingModel
from chiral_mols.model.classifier import ChiralityClassifier
from torch.optim import AdamW
from torch.nn import CrossEntropyLoss

dataset_dir = Path("/share/snw30/projects/chiral_mols/dataset/chiral_atoms")


training_cfg = TrainConfig(batch_size=6,learning_rate=1e-4, N_epochs= 100)
chiral_embedding_dim = 32 # Linear Projection from Pseudosclars to dim.
N_classes = 3

input_irreps = Irreps("128x0e+128x1o+128x0e")


dataset = PtrMoleculeDataset.reload_dataset_from_dir(dataset_dir)
dataset_splitting = DatasetSplitter(dataset.structure_ids)
train_idx, val_idx = dataset_splitting.random_split_by_molecule(
    train_val_ratios=[0.7, 0.3]
)
train_data = Subset(dataset, train_idx)
val_data = Subset(dataset, val_idx)

train_data_loader = DataLoader(train_data, batch_size= training_cfg.batch_size, collate_fn=concat_collate, shuffle=True,drop_last=True)
val_data_loader = DataLoader(val_data,batch_size = training_cfg.batch_size, collate_fn=concat_collate)

mean_inv, std_inv = get_mean_std_invariant_indices(train_data.dataset.embeddings, input_irreps)


chiral_embedding_model = ChiralEmbeddingModel(input_irreps=input_irreps, pseudoscalar_irreps= Irreps("128x0o"), output_embedding_dim=32, mean_inv_atomic_embedding= mean_inv, std_inv_atomic_embedding= std_inv)

classifier = ChiralityClassifier(chiral_embedding_dim=chiral_embedding_dim, hidden_dim= 64, n_classes = N_classes, dropout=0.1)

loss_fn = CrossEntropyLoss()
optimizer = AdamW(params = list(classifier.parameters())+ list(chiral_embedding_model.parameters()), lr = training_cfg.learning_rate )

device = "cuda"
classifier.to(device)
chiral_embedding_model.to(device)

for epoch in range(1, training_cfg.N_epochs+1):

    classifier.train()
    chiral_embedding_model.train()

    
    
    #train loop 
    accumulated_train_loss = 0.0
    for batch in train_data_loader:
        batch = batch.to_(device = device)

        optimizer.zero_grad()
        chiral_embedding = chiral_embedding_model(batch.embeddings)

        chiral_logits = classifier(chiral_embedding)

        loss = loss_fn(chiral_logits, batch.chirality_labels)

        loss.backward()
        optimizer.step()
        accumulated_train_loss += loss.item()
     

    
    # val loop
    with torch.no_grad():

        classifier.eval()
        chiral_embedding_model.eval()
        accumulated_val_loss = 0.0

        for batch in val_data_loader:
            batch.to_(device = device)
            chiral_embedding = chiral_embedding_model(batch.embeddings)

            chiral_logits = classifier(chiral_embedding)

            loss = loss_fn(chiral_logits, batch.chirality_labels)

            accumulated_val_loss += loss.item()


    print(f"Epoch: {epoch}. Train Loss: {accumulated_train_loss/len(train_data_loader)}, Validation Loss {accumulated_val_loss/len(val_data_loader)}")

    




# evaluation

# Mainly focus on the correct identification of chiral centers






