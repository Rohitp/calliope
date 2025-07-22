import torch
import torch.nn as nn


# GELU is like a fancy version of ReLU, it smooths out the activation function
# It also returns small negative values for negative inputs, instead of just zeroing them out
# So even negative inputs can contribute to the output, which helps with gradient flow
# Using the forumla here thats been approximated 
# See here -> https://datascience.stackexchange.com/questions/49522/what-is-gelu-activation
    

class GELU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        # Copied from https://datascience.stackexchange.com/questions/49522/what-is-gelu-activation
        # No questions asked
        return 0.5 * x * (1 + torch.tanh(torch.sqrt(torch.tensor(2) / torch.pi) * (x + 0.044715 * torch.pow(x, 3))))
    

# Feedforwars is a sequence of linear transformations with a non-linear activation function in between.
# Like a neural network.
# It takes the input expands to a larger space my multiplying it by 4 and contracts it again.
# It allows for richer representations
class FeedForward(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.layers = nn.Sequential(nn.Linear(config['emb_dim'], config['emb_dim'] * 4), GELU(), nn.Linear(config['emb_dim'] * 4, config['emb_dim']))

    def forward(self, x):
        return self.layers(x)
    

# indexes = indexes in current context, its shape is (batch, n_tokens)
# max_new tokens = number of tokens we want
# context_size = size of the context we're setting

def generate_text(model, indexes, max_new_tokens, context_size):
    for _ in range(max_new_tokens):

        # truncate context to the max context size
        indexes_cond = indexes[:, -context_size:]  # Get the last context_size tokens
        with torch.no_grad():
            logits = model(indexes_cond)
        
        logits = logits[:, -1, :]  # Get the logits for the last token

        # we don't need to apply softmax here, because we just want the most probable next token
        # did it anyway
        probabilities = torch.softmax(logits, dim=-1)

        # arg max returns the index of the highest probability token
        next = torch.argmax(probabilities, dim=-1, keepdim=True)  # Get the most probable next token
        indexes = torch.cat((indexes, next), dim=1) # Append the new token to the sequence
    return indexes
