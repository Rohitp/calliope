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