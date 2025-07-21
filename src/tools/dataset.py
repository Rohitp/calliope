import torch 
from torch.utils.data import Dataset

# https://pytorch.org/tutorials/beginner/basics/data_tutorial.html
# For easy iteration of the data
class CalliopeDataset(Dataset):

    # text -> full text that's being batched and loaded
    # tokenizer -> tokenizer to use for encoding. We use tiktoken here 
    # seq_length -> context window size 
    # stride -> size of the sliding window.
    def __init__(self, text, tokenizer, seq_length, stride):
        # Source and target is in a sliding window
        # For example, with -> Alice was beginning to get very tired of sitting
        # Alice --> was 
        # Alice was --> beginning
        # Alice was beginning --> to
        # Alice was beginning to --> get
        # and so on
        self.sources =[]
        self.targets = []
        tokens = tokenizer.encode(text)

        for i in range(0, len(tokens) - seq_length, stride):
            source = tokens[i : i+seq_length]
            target = tokens[i+1 : i+seq_length+1]
            self.sources.append(torch.tensor(source))
            self.targets.append(torch.tensor(target))
    
    def __len__(self):
        # It's unclear what len is supposed to override and why this is an immutabke field
        # Using this for inout sources
        return len(self.sources)

    def __getitem__(self, index):
        return self.sources[index], self.targets[index]

