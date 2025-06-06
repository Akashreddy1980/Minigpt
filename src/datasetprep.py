import torch

# Processing the data for the Minigpt model.
with open('../data/input.txt', 'r') as f:
    text = f.read()

chars = sorted(list(set(text)))
vocab_size = len(chars)
# print(chars)
asetn = {j: i for i, j in enumerate(chars)}
nseta = {i: j for j, i in asetn.items()}
# print(nseta)
#Simple character encoding and decoding functions as the tokenizer.
encode = lambda s: [asetn[c] for c in s]
decode = lambda l: ''.join([nseta[i] for i in l])

data = torch.tensor(encode(text), dtype=torch.long)
# print(data.shape)
# print(data[:100])  


# Splitting the data into training, validation, and test sets.
n1 = int(len(data) * 0.8)  # 80% for training       
n2 = int(len(data) * 0.9)  # 10% for validation

train_data = data[:n1]
val_data = data[n1:n2]
test_data = data[n2:]
torch.manual_seed(42)  # for reproducibility

def get_batch(split, batch_size=32, block_size=8):
    
    # Get a batch of data from the specified split (train, val, test).
    
    data = {
        'train': train_data,
        'val': val_data,
        'test': test_data
    }[split]

    ix = torch.randint(0, len(data) - block_size , (batch_size,))
    x_batch = torch.stack([data[i:i+block_size] for i in ix])
    y_batch = torch.stack([data[i+1:i+block_size+1] for i in ix])
    
    return x_batch, y_batch


# xbatch_test, ybatch_test = get_batch('test', batch_size=32, block_size=8)
if __name__ == "__main__":
    # Example usage
    xbatch_test, ybatch_test = get_batch('test', batch_size=32, block_size=8)
    print(xbatch_test.shape, ybatch_test.shape)

# print(xbatch_test.shape, ybatch_test.shape)

# print(xbatch_test)