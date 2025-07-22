# Dump file for random snippets I need to test 

import torch 
import torch.nn as nn
from tools.config import CALLIOPE_CONFIG_124M
from tools.utils import GELU
from modules.optimus import Optimus
from modules.polymnia import Polymnia
import tiktoken
from tools.utils import generate_text



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

layer_sizes = [3, 3, 3, 3, 3, 1]
sample_input = torch.tensor([[1., 0., -1.]])

# model_without_shortcut = SampleNN(layer_sizes, use_shortcut=False)
# torch.manual_seed(123)  # Reset seed for reproducibility
# model_with_shortcut = SampleNN(layer_sizes, use_shortcut=True)

# print("Testing model with shortcut connections:")
# test_gradients(model_with_shortcut, sample_input.clone().detach())
# print("Testing model without shortcut connections:")
# test_gradients(model_without_shortcut, sample_input.clone().detach())


tokenizer = tiktoken.get_encoding("gpt2")

batch = []

txt1 = "Every effort moves you"
txt2 = "Every day holds a"

batch.append(torch.tensor(tokenizer.encode(txt1)))
batch.append(torch.tensor(tokenizer.encode(txt2)))
batch = torch.stack(batch, dim=0)



start_context = "Hello, I am"

encoded = tokenizer.encode(start_context)
print("encoded:", encoded)

encoded_tensor = torch.tensor(encoded).unsqueeze(0)

torch.manual_seed(123)
model = Polymnia(CALLIOPE_CONFIG_124M)
# output = model(batch)
# print("Input batch:\n", batch)
# print("\nOutput shape:", output.shape)
# print(output)

# disable dropout for deterministic output
model.eval()

out = generate_text(model, encoded_tensor, max_new_tokens=6, context_size=CALLIOPE_CONFIG_124M["context_length"])

print("Output:", out)
print("Output length:", len(out[0]))

decoded_text = tokenizer.decode(out.squeeze(0).tolist())
print(decoded_text)

