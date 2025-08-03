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
    

# Feedforwars is a sequence of linear transformations with a non-linear activation function in between.
# Like a neural network.
# It takes the input expands to a larger space my multiplying it by 4 and contracts it again.
# It allows for richer representations
class FeedForward(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.layers = nn.Sequential(nn.Linear(config['emb_dim'], config['emb_dim'] * 4), GELU(), nn.Linear(config['emb_dim'] * 4, config['emb_dim']))

    def forward(self, x):
        return self.layers(x)
    

# indexes = indexes in current context, its shape is (batch, n_tokens)
# max_new tokens = number of tokens we want
# context_size = size of the context we're setting

def generate_text(model, indexes, max_new_tokens, context_size):
    for _ in range(max_new_tokens):

        # truncate context to the max context size
        indexes_cond = indexes[:, -context_size:]  # Get the last context_size tokens
        with torch.no_grad():
            logits = model(indexes_cond)
        
        logits = logits[:, -1, :]  # Get the logits for the last token

        # we don't need to apply softmax here, because we just want the most probable next token
        # did it anyway
        probabilities = torch.softmax(logits, dim=-1)


        # arg max returns the index of the highest probability token
        next = torch.argmax(probabilities, dim=-1, keepdim=True)  # Get the most probable next token
        indexes = torch.cat((indexes, next), dim=1) # Append the new token to the sequence
    return indexes



# Cross entropy loss is a measure of how well the model's predicted probabilities match the true labels
# it's the difference between two probability distributions, the inputs and the targets
# How you do this is 
# 1. You take some logits ->  logits = model(inputs)
# 2. Apply softmax and get probabilities -> probabilities = softmax(logits)
# 3. Get target probabilities -> targets = probabilities[target_indexes]
# 4. Convert them to log probabilities -> torch.log(targets)
# 5. Take the mean of it -> loss = torch.mean(log_probs)
# 6. Make it negative -> loss = -loss

# We try and reduce this negative value to 0 by backpropagation
# We just call the inbuilt function here

# Side note taking this one step further gives us perplexity 
# which is a measure of how well the model predicts the next token in the sequence
# it's torch.exp(loss).
# if perplexity is 45678, it means the model is unsure of which of the 45678 tokens to predict next
def cross_entropy_loss(input, target, model, device):
    input = input.to(device)
    target = target.to(device)
    logits = model(input)
    loss = nn.functional.cross_entropy(logits.flatten(0, 1), target.flatten())
    return loss



# This just iteates over all batches and accumulates the loss and returns it
def calc_loss_loader(dataloader, model, device, num_batches=None):
    total_loss = 0.0


    if(len(dataloader) == 0):
        return float('nan')
    
    # If batch size is not specified, we use the entire dataset
    elif num_batches is None:
        num_batches = len(dataloader)
    else:
        num_batches = min(num_batches, len(dataloader))


    for(i, (inputs, targets)) in enumerate(dataloader):
        if i < num_batches:
            loss = cross_entropy_loss(inputs, targets, model, device)
            total_loss += loss.item()
        else:
            break
        
    return total_loss / num_batches



# two functions to convert text to token ids and vice versa
# tired of always calling the logic seperately
def text_to_token_ids(text, tokenizer):
    encoded = tokenizer.encode(text, allowed_special={'<|endoftext|>'})
    encoded_tensor = torch.tensor(encoded).unsqueeze(0)
    return encoded_tensor

def token_ids_to_text(token_ids, tokenizer):
    flat = token_ids.squeeze(0)
    return tokenizer.decode(flat.tolist())