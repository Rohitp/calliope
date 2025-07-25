import torch 
from torch.utils.data import Dataset
import tiktoken

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
    


def create_dataloader(text, batch_size = 4, max_length = 256, stride = 128, shuffle = True, drop_last = True, num_workers = 0):

    tokenizer = tiktoken.get_encoding("gpt2")
    dataset = CalliopeDataset(text, tokenizer, max_length, stride)
    

    # batch_size is the number of tuples that we work on (seq_length is the number of tokens in each tuple)
    # shuffle randomises the order of the tuples to prevent overfitting. We don't want to shuffle, we want to preserve the order of the text
    # drop_last drops the last batch if it's not full 
    # num_workers is the number of threads to use for loading the data. 
    # Recommendation is to use os.cpu_count() / 2 for this.
    # All there parameters seem to be hyper specific and tweaking this to get results is paramount
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    return dataloader

