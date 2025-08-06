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
            output = f"\n\n###Response:\n{item['output']}"
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
def collate(batch, padding=50256, device='cpu', max_length=None, mask_value=-100):
    batch_max = max([len(item) for item in batch])
    inputs = []
    outputs = []

    for item in batch:
        padded = item + [padding] * (batch_max - len(item))
        inputs.append(torch.tensor(padded))
        # We do this to shift the input by one token to get the output
        padded += [padding] 
        output_temp = torch.tensor(padded[1:])

        mask = (output_temp == padding)

        # We need to mask everything but the first padding token
        # We do this so that the model does not learn to predict padding tokens
        # We still need one padding token to be predicted for an end of sentence
        # We choose -100 as a mask value that does not occur in the tokeniser.
        # Also this is used in the default cross_entropy function as a default thing to ignore -> https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.cross_entropy.html


        # Cumsum gives a cumulative number greater starting from 1 for padding. We use that with masked fill
        first_pad_only_mask = mask & (mask.cumsum(dim=0) == 1)
        final_mask = (output_temp != padding) | first_pad_only_mask
        masked_outputs = output_temp.masked_fill(~final_mask, mask_value)
        outputs.append(masked_outputs)





    input_result = torch.stack(inputs, dim=0).to(device)
    output_result = torch.stack(outputs, dim=0).to(device)
    return input_result, output_result



# Rewriting the implementation to compare different approaches as I suspect the first one is inefficient. Will compare and contrast
def collate_v2(batch, padding=50256, device='cpu', max_length=None, mask_value=-100):
    batch_max = max([len(item) + 1 for item in batch])
    inputs = []
    outputs = []

    for item in batch:
        new_item = item.copy()
        new_item += [padding]

        padded = (new_item + [padding] *(batch_max - len(new_item)))
        input_val = torch.tensor(padded[:-1])
        output_val = torch.tensor(padded[1:])


        mask = output_val == padding
        indices = torch.nonzero(mask).squeeze()

        if indices.numel() > 1:
            output_val[indices[1:]] = mask_value

        if max_length is not None:
            input_val = input_val[:max_length]
            output_val = output_val[:max_length]

        inputs.append(input_val)
        outputs.append(output_val)

    inputs_tensor = torch.stack(inputs).to(device)
    targets_tensor = torch.stack(outputs).to(device)
    return inputs_tensor, targets_tensor
