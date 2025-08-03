from turtle import reset
from src.tools.utils import calc_loss_loader, cross_entropy_loss, generate_text, text_to_token_ids, token_ids_to_text
import torch
import torch.nn as nn


# A simple training loop 

# iterate over all epochs 
#     iterate over batches in an epoch 
#         reset loss from previous batches
#         calculate loss on current batch 
#         backpropagate loss to get gradients
#         update model weights using the gradients
#     generate sample text for validation
# rinse and repeat

# Further reading for more topics -> https://stackoverflow.com/questions/55933867/what-does-learning-rate-warm-up-mean
# And this -> https://spotintelligence.com/2024/04/29/cosine-annealing-in-machine-learning/
# And this to work with learning rates -> https://neptune.ai/blog/understanding-gradient-clipping-and-how-it-can-fix-exploding-gradients-problem



# Every other parameter is either self explanatory or explained with comments in the flow but 

# eval_iter - number of batches 
# start_context - text to start generating from

def train_model(model, train_loader, val_loader, optimizer, device, num_epochs, eval_freq, eval_iter, start_context, tokenizer):


    train_losses, val_losses, track_tokens_seen = [], [], []
    tokens_seen, global_step = 0, -1


    # An epoch is a pass over the entire dataset, it seems every piece of training data through it
    # It's split into batches, which are smaller chunks of the dataset
    # Weights are updated after each batch, so the model learns incrementally

    for epoch in range(num_epochs):
        model.train()

        for input_batch, target_batch in train_loader:

            optimizer.zero_grad()
            loss = cross_entropy_loss(input_batch, target_batch, model, device)

            # Backpropagation to calculate gradients. Tells us how much each weight contributed to the loss
            loss.backward()

            # Updates model weights here
            # for param in model.parameters():
            #     param.data = param.data - learning_rate * param.grad
            # This is what it does to update the weights. The grad is a vector of the same shape as the weight tensor
            optimizer.step()

            tokens_seen += input_batch.numel()
            global_step += 1


            # We evaluate the model at a set frequency.
            # This is optional
            if global_step % eval_freq == 0:
                train_loss, val_loss = evaluate_model(model, train_loader, val_loader, device, eval_iter)
                train_losses.append(train_loss.item())
                val_losses.append(val_loss.item())
                track_tokens_seen.append(tokens_seen)
                print(f"Epoch {epoch+1}, Step {global_step}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Tokens Seen: {tokens_seen}")



        # Again optionally prints sample text after each epoch
        gen_sample(model, tokenizer, device, start_context)
            

    return train_losses, val_losses, track_tokens_seen





def evaluate_model(model, train_loader, val_loader, device, eval_iter):

    # We put it in eval mode to disable dropout and batch normalization
    model.eval()
    with torch.no_grad():
        train_loss = calc_loss_loader(train_loader, model, device, num_batches=eval_iter)
        val_loss = calc_loss_loader(val_loader, model, device, num_batches=eval_iter)

    # Switch back to training mode for next batch
    model.train()
    return train_loss, val_loss


def gen_sample(model, tokenizer, device, start_context):
    model.eval()
    context_size  = model.positional_embedding_layer.weight.shape[0]
    encoded = text_to_token_ids(start_context, tokenizer).to(device)
    with torch.no_grad():
        token_ids = generate_text(model, encoded, 50, context_size)

    decoded = token_ids_to_text(token_ids, tokenizer)
    print(f"Generated text: {decoded}")
    model.train()

