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