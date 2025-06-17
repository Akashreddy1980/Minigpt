import datasetprep
import torch
import torch.nn as nn
from torch.nn import functional as F
# import torch.nn.Functional as F
vocab_size = datasetprep.vocab_size
n_embd = 32
block_size = 8 
batch_size = 32
head_size = 8

class CharacterLevelModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.tokenEmbedding = nn.Embedding(vocab_size, n_embd)
        self.positionalEmbedding = nn.Embedding(block_size, n_embd)
        self.head = Head()
        self.lm_head = nn.Linear(head_size, vocab_size)  # Final projection to vocab size

    def forward(self, x, targets=None):
        B, T = x.shape
        tok_embd = self.tokenEmbedding(x)
        pos_embd = self.positionalEmbedding(torch.arange(T, device=x.device))
        x = tok_embd + pos_embd
        x = self.head(x)  # (B, T, head_size)
        logits = self.lm_head(x)  # (B, T, vocab_size)
        if targets is None:
            loss = None
        else:
            logits_flat = logits.view(B * T, vocab_size)
            loss = F.cross_entropy(logits_flat, targets.view(-1))
        return logits, loss
    
    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            # Only use the last block_size tokens as context
            idx_cond = idx[:, -block_size:]
            logits, loss = self(idx_cond)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx  
    
class Head(nn.Module):

    def __init__(self):
        super().__init__()
        self.Key = nn.Linear(n_embd,head_size,bias =False)
        self.Query = nn.Linear(n_embd,head_size, bias = False)
        self.Value = nn.Linear(n_embd,head_size, bias = False)
        self.tril = torch.tril(torch.ones(block_size,block_size))
    
    def forward(self,x):
        B,T,C = x.shape
        k = self.Key(x)
        q = self.Query(x)
        v = self.Value(x)
        wei = q @ k.transpose(-2,-1) * C**-0.5 # (B, T, head_size) @ (B, head_size, T) --> (B, T, T)
        # tril = torch.tril(torch.ones(T,T))
        wei = wei.masked_fill(self.tril[:T,:T] == 0, float('-inf'))
        wei = F.softmax(wei, dim = -1)
        wei = wei/ wei.sum(1,keepdim=True)
        out = wei @ v
        # print(out.shape)
        return out

class Multihead(nn.Module):
    def __init__(self, n_heads):
        super().__init__()
        self.heads = nn.ModuleList([Head() for _ in range(n_heads)])
    
    def forward(self, x):
        out = [head(x) for head in self.heads]
        return torch.cat(out, dim=-1)  # Concatenate outputs from all heads 

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, n_embd, num_heads, block_size):
        super().__init__()
        assert n_embd % num_heads == 0, "Embedding dim must be divisible by number of heads"
        self.num_heads = num_heads
        self.head_dim = n_embd // num_heads
        self.block_size = block_size

        self.key = nn.Linear(n_embd, n_embd, bias=False)
        self.query = nn.Linear(n_embd, n_embd, bias=False)
        self.value = nn.Linear(n_embd, n_embd, bias=False)
        self.proj = nn.Linear(n_embd, n_embd)
        self.tril = torch.tril(torch.ones(block_size, block_size))

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)   # (B, num_heads, T, head_dim)
        q = self.query(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2) # (B, num_heads, T, head_dim)
        v = self.value(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2) # (B, num_heads, T, head_dim)

        att = (q @ k.transpose(-2, -1)) * (self.head_dim ** -0.5)  # (B, num_heads, T, T)
        att = att.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        out = att @ v  # (B, num_heads, T, head_dim)
        out = out.transpose(1, 2).contiguous().view(B, T, C)  # (B, T, n_embd)
        out = self.proj(out)
        return out

n = CharacterLevelModel()  # Initialize the model with vocabulary size

# x = h.forward(torch.randn(batch_size, block_size, vocab_size))
# print(x.shape)

idx = torch.zeros((1, 1), dtype=torch.long)  # Initialize input with a single token
#Inference generation.
# print(datasetprep.decode((n.generate(idx,max_new_tokens=100)[0]).tolist()))
# print(len((n.generate(idx,max_new_tokens=100)[0]).tolist()))

#implementation of an optimizer

optimizer = torch.optim.AdamW(n.parameters(), lr = 0.01)

for i in range(50000):

    xb,yb = datasetprep.get_batch("train")
    logits,loss = n(xb,yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

print(loss.item())


print(datasetprep.decode((n.generate(idx,max_new_tokens=1000)[0]).tolist()))