import torch 
from torch.utils.data import Dataset
from train.train_utils import finetune_helper


# Custom class for training the data.

class TrainData(Dataset):
    def __init__(self, data, tokenizer):
        self.data = data
        self.encoded = []

        for item in data:
            instructions = finetune_helper(item)
            output = f"\n\n###Response:\n{item['response']}"
            text = instructions + output
            self.encoded.append(tokenizer.encode(text))


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.encoded[idx]
    


# 50256 is the endoftext token in tiktoken. We use that as adding directly
# What we do is we take a list of inputs 
# inputs_1 = [0, 1, 2, 3, 4]
# inputs_2 = [5, 6]
# inputs_3 = [7, 8, 9]
# pad it with the endoftext token or its ID which is 50256
# so all batches have the same size, stack and return it. 
# Each bach can have its own max size. This way we can chunk and use them individually
def collate(batch, padding=50256, device='cpu'):
    batch_max = max([len(item) for item in batch])
    inputs = []

    for item in batch:
        padded = item + [padding] * (batch_max - len(item))
        inputs.append(torch.tensor(padded))

    result = torch.stack(inputs, dim=0).to(device)
    return result
