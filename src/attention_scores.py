import torch.nn as nn
import torch

# Calculating attention scores
# After the positoninal embedding dictate the relative positions of the words
# Attention scores dictate importance
class AttentionScores(nn.Module):
    def __init__(self, d_in, d_out, qkv_bias=False):
        super().__init__()
        # Query -> The thing we're querying or using as a reference
        # key is like a database key we look up the reference to 
        # Value is the result as in a key value pair
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)

    def forward(self, inputs):
        queries = self.W_query(inputs)
        keys = self.W_key(inputs)
        values = self.W_value(inputs)
        attention_scores = queries @ keys.T
        attention_weights = torch.softmax(attention_scores/keys.shape[-1]**0.5, dim=-1)
        context_vector = attention_weights @ values
        return context_vector