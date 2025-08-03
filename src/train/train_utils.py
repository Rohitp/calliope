from src.tools.utils import calc_loss_loader, cross_entropy_loss, generate_text, text_to_token_ids, token_ids_to_text
import torch
import torch.nn as nn

def train_model(model, train_loader, val_loader, optimizer, device, num_epochs, eval_freq, eval_iter, start_context, tokenizer):


    train_losses, val_losses, track_tokens_seen = [], [], []
    tokens_seen, global_step = 0, -1

    for epoch in range(num_epochs):
        model.train()

        for input_batch, target_batch in train_loader:

            optimizer.zero_grad()
            loss = cross_entropy_loss(input_batch, target_batch, model, device)
            loss.backward()
            optimizer.step()

            tokens_seen += input_batch.numel()
            global_step += 1

            if global_step % eval_freq == 0:
                train_loss, val_loss = evaluate_model(model, train_loader, val_loader, device, eval_iter)
                train_losses.append(train_loss.item())
                val_losses.append(val_loss.item())
                track_tokens_seen.append(tokens_seen)
                print(f"Epoch {epoch+1}, Step {global_step}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Tokens Seen: {tokens_seen}")


        gen_sample(model, tokenizer, device, start_context)
            

    return train_losses, val_losses, track_tokens_seen



def evaluate_model(model, train_loader, val_loader, device, eval_iter):
    model.eval()
    with torch.no_grad():
        train_loss = calc_loss_loader(train_loader, model, device, num_batches=eval_iter)
        val_loss = calc_loss_loader(val_loader, model, device, num_batches=eval_iter)
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

