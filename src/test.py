# Dump file for random snippets I need to test 

import torch 
from attention_scores import AttentionScores


test = torch.rand(3, 5)

test1 = torch.arange(5)
test3 = torch.ones(5)


inputs = torch.tensor([
    [0.43, 0.15, 0.89], 
    [0.55, 0.87, 0.66], 
    [0.57, 0.85, 0.64], 
    [0.22, 0.58, 0.33], 
    [0.77, 0.25, 0.10], 
    [0.05, 0.80, 0.55]
])

# A dot product can be considered as a measure of similarity between two vectors. 
# The higher the dot product, the more similar the two vectors are.
# print(torch.dot(test1, test1))


# Softmax normalises vectord and makes them sum up to 1
# This also ensures the values are always positive so more useful as probabilities
# And a mesure of relative importance
# print(torch.softmax(test, dim=0))


x_2 = inputs[1]
weighted_dot = []
d_in = inputs.shape[1]
print(d_in)
d_out = 2

torch.manual_seed(123)
W_query = torch.nn.Parameter(torch.randn(d_in, d_out), requires_grad=False)
W_key = torch.nn.Parameter(torch.randn(d_in, d_out), requires_grad=False)
W_value = torch.nn.Parameter(torch.randn(d_in, d_out), requires_grad=False)


query_2 = x_2 @ W_query
key_2 = x_2 @ W_key
value_2 = x_2 @ W_value


queries = inputs @ W_query
keys = inputs @ W_key
values = inputs @ W_value


attention_scores_2 = query_2 @ keys.T
key_dimensions = keys.shape[-1]

attn_weights_2 = torch.softmax(attention_scores_2/key_dimensions**0.5, dim=-1)

torch.manual_seed(789)
attention_scores = AttentionScores(d_in, d_out)

queries = attention_scores.W_query(inputs)
keys = attention_scores.W_key(inputs)
attention_scores = queries @ keys.T
attention_weights = torch.softmax(attention_scores/key_dimensions**0.5, dim=-1)
context_length = attention_scores.shape[0]
mask = torch.tril(torch.ones(context_length, context_length))
masked_weights = attention_weights * mask
attention_weights = torch.softmax(masked_weights/key_dimensions**0.5, dim=-1)




mask = torch.triu(torch.ones(context_length, context_length), diagonal=1)
masked = attention_scores.masked_fill(mask.bool(), -torch.inf)
print(masked)









