# Like all my python scripts, this main file is going to become an unstructured mess
# Will refactor it later


import re
from tokenizer import Tokenizer
from dataset import CalliopeDataset
import tiktoken
from torch.utils.data import DataLoader



BATCH_SIZE = 5


with open("../text/alice.txt", "r") as f:
    calliope = f.read()



test = "Hello “Well!” thought Alice to herself, “after such a fall as this, I shall think nothing of tumbling down stairs! Ooga"

# Splitting on whitespace and punctuation
# Copied from deepseek. I've forgotten regular expressions. Seems to work good enough
pattern = r'([,.:;?_!"()\']|--|\s)'
tokens = re.split(pattern, calliope)

text = [token.strip() for token in tokens if token.strip()]
vocabulary = sorted(set(text))
vocabulary.extend(set(["<|endoftext|>", "<|unk|>"]))


# A set of all unique lexemes in the text
lexicon = {word: i for i, word in enumerate(vocabulary)}
to_int = {i : word for i, word in lexicon.items()}
for i, word in enumerate(to_int.items()):
    if i < 10:
        # print(f"{i}: {word}")
        pass
    else:
        break




for i in range(1, 10):
    print("x: "+tokenizer.decode(encoded[:i]))
    print("y: "+tokenizer.decode([encoded[i]]))



def load_data(text, batch_size, seq_length, stride):
    tokenizer = tiktoken.get_encoding("gpt2")
    dataset = CalliopeDataset(text, tokenizer, seq_length, stride)

    # batch_size is the number of tuples that we work on (seq_length is the number of tokens in each tuple)
    # shuffle randomises the order of the tuples to prevent overfitting
    # drop_last drops the last batch if it's not full 
    # num_workers is the number of threads to use for loading the data
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=4)
    return dataloader
