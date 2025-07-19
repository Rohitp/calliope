# Smallest GPT-2 model is 124M -> https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf

CALLIOPE_CONFIG_124M = {
"vocab_size": 50257, # Vocabulary size, used by tiktoken as seen
"context_length": 1024, # Context length, max number of input tokens
"emb_dim": 768, # Embedding dimension
"n_heads": 12, # Number of attention heads
"n_layers": 12, # Number of layers
"drop_rate": 0.1, # Dropout rate
"qkv_bias": False # Query-Key-Value bias, seems like this is always false in most modern architectures.
}