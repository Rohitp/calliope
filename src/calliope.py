# Like all my python scripts, this main file is going to become an unstructured mess
# Will refactor it later


import re
from tokenizer import Tokenizer
from dataset import CalliopeDataset
import tiktoken
import torch
from torch.utils.data import DataLoader



# To form an embedding matrix from
# The embedding dimension is the size of the tensor that we use to represent each token
# Can pump up the embedding dimension to 512 or 1024. For reference, the original GPT-2 model used 768
# Make sure EMBEDDING_DIM % NUM_HEADS == 0
# NUM_HEADS is the number of attention heads, essentially the number of parallel attention mechanisms that we use to process the input
VOCAB_SIZE = tiktoken.get_encoding("gpt2").n_vocab
EMBEDDING_DIM = 256
NUM_HEADS = 0


# Hyperparameters
# WINDOW_SIZE = SEQ_LENGTH -> Wasted tokens at the end, but fastest
# WINDOW_SIZE = 1 -> No wasted data, but slowest
# WINDOW_SIZE = SEQ_LENGTH / 2 -> Middle ground
# BATCH_SIZE is the number of sample process at once
# SEQ_LENGTH is the number of tokens in each sample.
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

# print(token_embedding.shape)
# print(positional_embedding.shape)
# print(input_embedding.shape)


# Defining self attention vectors to indicate the importance of each token in the sequence
# For each token, a self attention vector is like this
# tokens Hello -> [.678, .123, .456]
# tokens world -> [.456, .789, .123]
# tokens cat -> [.123, .456, .789]
# you take each token and multiply it with the entire corupus, which is the full matrix. 
# This gives you a scalar for each token
# A a tensor containing all the token scalars
# A dot product can be considered as a measure of similarity between two vectors. 
# The higher the dot product, the more similar the two vectors are.
# Code like this
# ============================================
# weighted_dot = []
# for i in inputs:
#     weighted_dot.append(torch.dot(vec, i)) 
# print(torch.tensor(weighted_dot))
# ============================================

# Then we Softmax normalises vectord and makes them sum up to 1
# This also ensures the values are always positive so more useful as probabilities
# And a mesure of relative importance
# ============================================
# normalised = torch.softmax(torch.tensor(weighted_dot), dim=0)
# ============================================
# Finally the context vector is the weight sum of the product of the normalised weights and the input vectors
# ============================================
# context_vector = torch.zeros(vec.shape)
# for i, vector in enumerate(inputs):
#     context_vector+= normalised[i] * vector
# ============================================
# We simplify these operations as such -> Attention weights is just multuplying the full matrix with it's transpose

weighted_attention_scores = input_embedding @ input_embedding.T
normalised_attention_scores = torch.softmax(weighted_attention_scores, dim=-1)
context_vector = normalised_attention_scores @ input_embedding

# Stupid three lines of code took an evening to fully deep dive and understand