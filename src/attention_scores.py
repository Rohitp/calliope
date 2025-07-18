import torch.nn as nn
import torch

# Calculating attention scores
# After the positoninal embedding dictate the relative positions of the words
# Attention scores dictate importance
class AttentionScores(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout=0.0, qkv_bias=False):
        super().__init__()
        # Query -> The thing we're querying or using as a reference
        # key is like a database key we look up the reference to 
        # Value is the result as in a key value pair
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)


        # A dropout randomly drops a percentage of teh attention weights, this prevents overfitting and is used only udring training
        # Another thing to note is that if say the dropout rate is .5, 50% of weights become 0, but the other 50% of weights are scaled to
        # Double the value to compensate for the missing weights
        self.dropout = nn.Dropout(dropout)

        # Not strictly needed. Moves buffers to the righr device - CPU vs GPU automatically
        self.register_buffer("mask", torch.triu(torch.ones(context_length, context_length), diagonal=1))

    def forward(self, inputs):

        b, num_tokens, d_in = inputs.shape

        queries = self.W_query(inputs)
        keys = self.W_key(inputs)
        values = self.W_value(inputs)


        # Swapping only batch_size and seq_length
        attention_scores = queries @ keys.transpose(1,2)

        # So, a couple of things going on here.
        # 1. We mask the attention scores to prevent looking at future tokens in the sequence. This is important for autoregressive models like GPT.
        #     for if the input is my name is baasha, maanick baasha, when looking at name we want to zero out everything after.
        # 2. We can create a lower triangular matrix where the upper triangel is -inf so we can apply softmax. 0 doesnt work as e^0 == 1
        attention_scores.masked_fill_(self.mask.bool()[:num_tokens, :num_tokens], -torch.inf)

        # We use normalise by key dimension to make the attention scores more stable
        attention_weights = torch.softmax(attention_scores/keys.shape[-1]**0.5, dim=-1)
        context_vector = attention_weights @ values
        return context_vector