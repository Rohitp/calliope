# Smallest GPT-2 model is 124M -> https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf

CALLIOPE_CONFIG_124M = {
"vocab_size": 50257, # Vocabulary size, used by tiktoken as seen, the output of the final transformer block is a logits vector of this size
"context_length": 256, # Context length, max number of input tokens, we set it to 256 for training locally, set it back to 1024 
"emb_dim": 768, # Embedding dimension
"n_heads": 12, # Number of attention heads
"n_layers": 12, # Number of layers, this is also the number of transformer blocks
"drop_rate": 0.1, # Dropout rate
"qkv_bias": False, # Query-Key-Value bias, seems like this is always false in most modern architectures.
"learning_rate": 0.001, # Learning rate. Need to change this later to somethig like 3e-4
"weight_decay": 0.1, # Weight decay for AdamW optimizer, change it later to something like 1e-1
"temperature": 1.4, # Temperature -> higher temperature means more random, lower temperature means more deterministic
"top_k": 25, # Top-k sampling -> limits the number of samples you pick
}