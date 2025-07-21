# Dump file for random snippets I need to test 

import torch 
import torch.nn as nn
import tiktoken
from modules.attention_scores import AttentionScores
from config import CALLIOPE_CONFIG_124M
from modules.polymnia import Polymnia, PolymniaTransformerBlock, PolymniaLayerNorm




inputs = torch.tensor([
    [0.43, 0.15, 0.89], 
    [0.55, 0.87, 0.66], 
    [0.57, 0.85, 0.64], 
    [0.22, 0.58, 0.33], 
    [0.77, 0.25, 0.10], 
    [0.05, 0.80, 0.55]
])





tokenizer = tiktoken.get_encoding("gpt2")
batch = []
for i in ["Every effort moves you", "every day holds a"]:
    batch.append( torch.tensor(tokenizer.encode(i)))
batch = torch.stack(batch, dim=0)

torch.manual_seed(123)
# model = Polymnia(CALLIOPE_CONFIG_124M)
# logits = model(batch)


batch_example = torch.randn(2, 5)
layer = nn.Sequential(nn.Linear(5, 6), nn.ReLU())



out = layer(batch_example)
print(out)


ln = PolymniaLayerNorm(emb_dim=5)
output = ln(batch_example)
mean_p = output.mean(dim=-1, keepdim=True)
var_p = output.var(dim=-1, unbiased=False, keepdim=True)
print(mean_p)
print(var_p)
