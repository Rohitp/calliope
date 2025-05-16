# Like all my python scripts, this main file is going to become an unstructured mess
# Will refactor it later


import re
from tokenizer import Tokenizer
from dataset import CalliopeDataset
import tiktoken
import torch
from torch.utils.data import DataLoader



# To form an embedding matrix from
# The e,mbedding dimension is the size of the tensor that we use to represent each token
# Can pump up the embedding dimension to 512 or 1024. For reference, the original GPT-2 model used 768
# Make sire EMBEDDING_DIM % NUM_HEADS == 0
# NUM_HEADS is the number of attention heads, essentially the number of parallel attention mechanisms that we use to process the input
VOCAB_SIZE = tiktoken.get_encoding("gpt2").n_vocab
EMBEDDING_DIM = 256
NUM_HEADS = 0


# Hyperparameters
# WINDOW_SIZE = SEQ_LENGTH -> Wasted tokens at the end, but fastest
# WINDOW_SIZE = 1 -> No wasted data, but slowest
# WINDOW_SIZE = SEQ_LENGTH / 2 -> Middle ground
BATCH_SIZE = 8
SEQ_LENGTH = 4
WINDOW_SIZE = 4


# BATCH_SIZE = 1
# SEQ_LENGTH = 4
# WINDOW_SIZE = 1

# Splitting on whitespace and punctuation
# Copied from deepseek. I've forgotten regular expressions. Seems to work good enough
SPLIT_REGEX = r'([,.:;?_!"()\']|--|\s)'


with open("../text/alice.txt", "r") as f:
    calliope = f.read()



tokens = re.split(SPLIT_REGEX, calliope)

text = [token.strip() for token in tokens if token.strip()]
vocabulary = sorted(set(text))

# Extending by adding the special tokens. 
# eot seperates different corpus sources. Unkown is for tokens not encountered before
vocabulary.extend(set(["<|endoftext|>", "<|unk|>"]))


# A set of all unique lexemes in the text
lexicon = {word: i for i, word in enumerate(vocabulary)}


tokenizer = tiktoken.get_encoding("gpt2")




def load_data(text, tokenizer, batch_size, seq_length, window_size):
    
    dataset = CalliopeDataset(text, tokenizer, seq_length, window_size)

    # batch_size is the number of tuples that we work on (seq_length is the number of tokens in each tuple)
    # shuffle randomises the order of the tuples to prevent overfitting. We don't want to shuffle, we want to preserve the order of the text
    # drop_last drops the last batch if it's not full 
    # num_workers is the number of threads to use for loading the data. 
    # Recommendation is to use os.cpu_count() / 2 for this.
    # All there parameters seem to be hyper specific and tweaking this to get results is paramount
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=True, num_workers=0)
    return dataloader



dataloader = load_data(calliope, tokenizer, BATCH_SIZE, SEQ_LENGTH, WINDOW_SIZE)


iterable = iter(dataloader)
source, targets = next(iterable)

print(source)

token_embedding_layer = torch.nn.Embedding(VOCAB_SIZE, EMBEDDING_DIM)

# This is simply a loopup that maps the source token IDs to the embedding layer
token_embedding = token_embedding_layer(source)



# GPT2 uses an assolute positioning layer to save positioning information
# This is a standard multiplier that is applied to the token embedding
# Without this, [hello, world] and [world, hello] would be the same
# Or more weirdly, the cat sat and cat the sat would be same
# https://www.lesswrong.com/posts/qvWP3aBDBaqXvPNhS/gpt-2-s-positional-embedding-matrix-is-a-helix

positional_embedding_layer = torch.nn.Embedding(SEQ_LENGTH, EMBEDDING_DIM)
positional_embedding = positional_embedding_layer(torch.arange(SEQ_LENGTH))

input_embedding = token_embedding + positional_embedding

print(token_embedding.shape)
print(positional_embedding.shape)
print(input_embedding.shape)
