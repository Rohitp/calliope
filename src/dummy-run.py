# Dump file for random snippets I need to test 

import torch 
import torch.nn as nn
from tools.config import CALLIOPE_CONFIG_124M
from tools.utils import GELU
from modules.optimus import Optimus
from modules.polymnia import Polymnia
import tiktoken
from tools.utils import generate_text, text_to_token_ids, token_ids_to_text
from train.train_utils import finetune_helper
import json

torch.manual_seed(123)





class SampleNN(nn.Module):
    def __init__(self, layer_sizes, use_shortcut):
        super().__init__()
        self.use_shortcut = use_shortcut
        self.layers = nn.ModuleList([
            nn.Sequential(nn.Linear(layer_sizes[0], layer_sizes[1]), GELU()),
            nn.Sequential(nn.Linear(layer_sizes[1], layer_sizes[2]), GELU()),
            nn.Sequential(nn.Linear(layer_sizes[2], layer_sizes[3]), GELU()),
            nn.Sequential(nn.Linear(layer_sizes[3], layer_sizes[4]), GELU()),
            nn.Sequential(nn.Linear(layer_sizes[4], layer_sizes[5]), GELU())
            ])

    def forward(self, x):
        for layer in self.layers:
            output = layer(x)

            # Shortcut here doesn't mean skip connection, it means to not just pass the output
            # but to keep the input as well
            # we only shortcut when the input and output shapes match
            if self.use_shortcut and x.shape == output.shape:
                x = x + output
            else:
                x = output
        return x
    


def test_gradients(model, input):


    output = model(input)
    print("Output:", output)

    target = torch.tensor([[0.]])
    loss = nn.MSELoss()(output, target)
    loss.backward()

    for name, param in model.named_parameters():
        if 'weight' in name:
            print(f"Gradient for {name}: {param.grad.abs().mean().item()}")

def reset_weights(model):
    for layer in model.modules():
        if isinstance(layer, nn.Linear):
            layer.reset_parameters()






