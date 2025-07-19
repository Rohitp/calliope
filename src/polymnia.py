import torch
import torch.nn as nn


class Polymnia(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.token_embedding_layer = nn.Embedding(config['vocab_size'], config['emb_dim'])
        self.positional_embedding_layer = nn.Embedding(config['context_length'], config['emb_dim'])
        self.dropout = nn.Dropout(config['drop_rate'])

        self.transformer_blocks = nn.Sequential([ 1 for _ in range(config['n_layers'])])

        self.final_layer_norm = nn.LayerNorm(config['emb_dim'])
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
        logits = self.out_head(var)
        return logits
