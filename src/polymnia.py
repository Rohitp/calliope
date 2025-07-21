import torch
import torch.nn as nn


# Define a basic GPT model here

class Polymnia(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.token_embedding_layer = nn.Embedding(config['vocab_size'], config['emb_dim'])
        self.positional_embedding_layer = nn.Embedding(config['context_length'], config['emb_dim'])
        self.dropout = nn.Dropout(config['drop_rate'])

        self.transformer_blocks = nn.Sequential(*[PolymniaTransformerBlock(config) for _ in range(config['n_layers'])])

        self.final_layer_norm = PolymniaLayerNorm(config['emb_dim'])
        self.out_head = nn.Linear(config['emb_dim'], config['vocab_size'], bias=False)

      
        

        # forward always defines how computations flow through the model, or a forward pass if you may
        # it's called implicitly like ClassName(inputs) or model(inputs), it does __call__ under the hood
        # It does three things
        # 1. Sets up hooks for debugging
        # 2. setsup autograd for backpropagation
        # 3. Calls the logic in the forward function
    def forward(self, x):
        batch_size, seq_length = x.shape
        token_embedding = self.token_embedding_layer(x)
        positional_embedding = self.positional_embedding_layer(torch.arange(seq_length, device=x.device))

        var = token_embedding + positional_embedding
        var = self.dropout(var)
        var = self.transformer_blocks(var)
        var = self.final_layer_norm(var)


        # raw scores = logits. Converted to probabilites later
        logits = self.out_head(var)
        return logits
    

class PolymniaTransformerBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
 

    def forward(self, x):
        return x
    


# Layer normalization is taking the values of the layer and normalizing them to have a mean of 0 and a standard deviation of 1.
# This helps stablise the process as values diverge freatly over many layers
# And this helps keep results predictable
# For example take weights of -> [0.22, 0.34, 0.00, 0,22, 0,00] -> mean: 0.13, variance: 0.39
# Normalise to -> [0.61, 1.41, -0.87, 0.58, -0.87, -0.87] -> mean: 0, variance: 1
class PolymniaLayerNorm(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        # Very small value to prevent division by zero. This is so small that it doesn't affect the results
        self.epsilon = 1e-5
        # These are learnable parameters that scale and shift the normalized values
        # During training we train these as well to best adjust values
        # GPT architecture uses them, so do we
        self.scale = nn.Parameter(torch.ones(emb_dim))
        self.shift = nn.Parameter(torch.zeros(emb_dim))
    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)

        # There is a very deep rabbit hole here, for example
        # https://en.wikipedia.org/wiki/Bessel%27s_correction 
        # if unbiased is set to true this applyes bessesl correction, which also slightly increases mean square error
        # and bessels correction seems statistically insignificant at these levels
        # also I'm not sure if this will be fully gpt2 compatible otherwise.
        # This has taken me way too long to get this one line and therefore I will leave it with my level of understanding
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        normalised = (x - mean) / torch.sqrt(var + self.epsilon)
        return  self.scale * normalised + self.shift
