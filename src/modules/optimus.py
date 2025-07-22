# okay, I started with classy and elegant greek muses for names
# but when it came time to name the transformer, I just went with OPTIMUS PRIME

import torch.nn as nn
from modules.attention_scores import AttentionScores
from tools.normaliser import PolymniaLayerNorm
from tools.utils import FeedForward


# A transformeer combines multi head attention with a feed forward network.
# It also adds dropout and shortcutting. 
# The operations in a transformer preserve the dimensionality of the input. (in this case 768)

# Self attention analyses the relationships between tokens in a sequence.
# FeedForward processes each token independently.


class Optimus(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attention = AttentionScores(d_in=config["emb_dim"], d_out=config["emb_dim"], context_length=config["context_length"], num_heads=config["n_heads"], dropout=config["drop_rate"], qkv_bias=config["qkv_bias"])
        self.feed_forward = FeedForward(config)
        self.layer_norm1 = PolymniaLayerNorm(config["emb_dim"])
        self.layer_norm2 = PolymniaLayerNorm(config["emb_dim"])
        self.drop_shortcut = nn.Dropout(config["drop_rate"])

    def forward(self, x):

        # Shortcut here doesn't mean skip connection, it means to not just pass the output
        # but to keep the input as well
        # we only shortcut when the input and output shapes match

        # Here in the forward pass we apply attention, and feed forward
        # Layer normalisation is applied before and dropout is applied after
        # This is like original GPT2 implementation
        shortcut = x
        x = self.layer_norm1(x)
        x = self.attention(x)
        x = self.drop_shortcut(x)
        x = shortcut + x  # Shortcut connection

        shortcut = x
        x = self.layer_norm2(x)
        x = self.feed_forward(x)
        x = self.drop_shortcut(x)
        x = shortcut + x  # Shortcut connection
        return x