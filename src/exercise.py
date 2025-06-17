
import torch
import torch.nn.functional as F
import torch.nn as nn

#practice implementation of self attention 
#we need a way model can also see and consider the previous token and easiest way is to have the averages of previous tokens.
torch.manual_seed(1234)
B = 3
T = 3
C = 3
x = torch.randn(B,T,C)

# print(x)
xbow = torch.zeros(B,T,C)
for b in range(B):
    for t in range(T):
        xprev = x[b,:t+1]
        xbow[b,t] = torch.mean(xprev,0)
        # print(torch.mean(xprev,0))

# print(xbow)
head_size = 16

Key = nn.Linear(C,head_size,bias =False)
Query = nn.Linear(C,head_size, bias = False)
Value = nn.Linear(C,head_size, bias = False)
k = Key(x)
q = Query(x)
v = Value(x)
wei = q @ k.transpose(-2,-1) * head_size**-0.5 # (B, T, head_size) @ (B, head_size, T) --> (B, T, T)
tril = torch.tril(torch.ones(T,T))
wei = wei.masked_fill(tril == 0, float('-inf'))
wei = F.softmax(wei, dim = -1)
wei = wei/ wei.sum(1,keepdim=True)
out = wei @ v

# print(xbow2)
print(torch.arange(T).shape)
