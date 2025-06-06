import datasetprep
import torch
import torch.nn as nn
from torch.nn import functional as F
# import torch.nn.Functional as F

class CharacterLevelModel(nn.Module):
    
    def __init__(self, vocab_size):
        super().__init__()
        self.tokenEmbedding = nn.Embedding(vocab_size, vocab_size)  # Embedding layer

    def forward(self, x, targets=None):

        if targets is None:
            loss = None
            logits = self.tokenEmbedding(x)
        else:
            logits = self.tokenEmbedding(x)  # Forward pass through the embedding layer
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            loss = F.cross_entropy(logits, targets.view(-1))  # Compute the loss
        return logits, loss
    
    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            logits, loss = self(idx)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)   # Apply softmax to get probabilities
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1) # Append the new token to the sequence
        return idx  
    

n = CharacterLevelModel(datasetprep.vocab_size)  # Initialize the model with vocabulary size
# x, y = datasetprep.get_batch('train', batch_size=32, block_size=8)  # Get a test batch

# print(x.shape, y.shape)  

# idx, loss = n(x, y)  # Forward pass through the model
# print(idx.shape, loss.item())

idx = torch.zeros((1, 1), dtype=torch.long)  # Initialize input with a single token
#Inference generation.
print(datasetprep.decode((n.generate(idx,max_new_tokens=100)[0]).tolist()))