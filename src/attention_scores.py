import torch.nn as nn
import torch

# Calculating attention scores
# After the positoninal embedding dictate the relative positions of the words
# Attention scores dictate importance

# Checkout -> https://arxiv.org/abs/1706.03762 for multi head reference
class AttentionScores(nn.Module):
    def __init__(self, d_in, d_out, context_length, num_heads, dropout=0.0, qkv_bias=False):
        super().__init__()


        assert d_out % num_heads == 0, "d_out must be divisible by num_heads"
        self.d_out = d_out
        self.num_heads = num_heads

        # Key operation for multi-head attention
        # We split the output dimension into multiple heads here
        self.head_dim = d_out // num_heads
        


        # Query -> The thing we're querying or using as a reference
        # key is like a database key we look up the reference to 
        # Value is the result as in a key value pair
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)



        # The output projection is used to combine the outputs of the attention heads
        self.out_projection = nn.Linear(d_out, d_out)


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


        # Reshape the queries, keys and values to have multiple heads
        # Basically splitting the d_out dimension into num_heads and head_dim
        # We transpose the dimensions so that we can do matrix multiplication
        # Ultimately this is just a really fancy way of looping around the single head in a wrapper class.
        # Infact it would have been better code to just do 

        # ===============================================
        # self.heads = [AttentionScores(d_in, self.head_dim, context_length, 1, dropout, qkv_bias) for _ in range(self.num_heads)]
        # ================================================


        # and in the forward function do
        # ============================================
        # torch.cat([head(x) for head in self.heads], dim=-1)
        # ============================================
        # This is more efficient, and it allows us to do one matrix multiplication like queries = self.W_query(inputs)
        # and not do it in a loop
        # but how noticeably, I need to experiment, but from what I undertand the gains should be major

        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)
        keys = keys.view(b, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)
        values = values.view(b, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)


        # Swapping only batch_size and seq_length
        attention_scores = queries @ keys.transpose(2,3)

        # So, a couple of things going on here.
        # 1. We mask the attention scores to prevent looking at future tokens in the sequence. This is important for autoregressive models like GPT.
        #     for if the input is my name is baasha, maanick baasha, when looking at name we want to zero out everything after.
        #.    honestly, why we do this and why can't bidrectional models like BERT be used for text generation is beyond me
        # 2. We can create a lower triangular matrix where the upper triangel is -inf so we can apply softmax. 0 doesnt work as e^0 == 1
        attention_scores.masked_fill_(self.mask.bool()[:num_tokens, :num_tokens], -torch.inf)

        # We use normalise by key dimension to make the attention scores more stable
        attention_weights = torch.softmax(attention_scores/keys.shape[-1]**0.5, dim=-1)


        # We apply dropout to the attention weights
        attention_weights = self.dropout(attention_weights)

        context_vector = (attention_weights @ values).transpose(1, 2)
        context_vector = context_vector.contiguous().view(b, num_tokens, self.d_out)
        context_vector = self.out_projection(context_vector)
        return context_vector