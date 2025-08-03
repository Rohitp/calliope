import numpy as np
from tools.utils import assign


# Just copying assigments one by one. I swear I'll comment this later
# This is more annoying than building the model

def load_weights_into_gpt(gpt, params):
    gpt.positional_embedding_layer.weight = assign(gpt.positional_embedding_layer.weight, params['wpe'])
    gpt.token_embedding_layer.weight = assign(gpt.token_embedding_layer.weight, params['wte'])

    for b in range(len(params["blocks"])):
        q_w, k_w, v_w = np.split((params["blocks"][b]["attn"]["c_attn"])["w"], 3, axis=-1)
        gpt.transformer_blocks[b].attention.W_query.weight = assign(gpt.transformer_blocks[b].attention.W_query.weight, q_w.T)
        gpt.transformer_blocks[b].attention.W_key.weight = assign(gpt.transformer_blocks[b].attention.W_key.weight, k_w.T)
        gpt.transformer_blocks[b].attention.W_value.weight = assign(gpt.transformer_blocks[b].attention.W_value.weight, v_w.T)

        q_b, k_b, v_b = np.split((params["blocks"][b]["attn"]["c_attn"])["b"], 3, axis=-1)

        gpt.transformer_blocks[b].attention.W_query.bias = assign(gpt.transformer_blocks[b].attention.W_query.bias, q_b)
        gpt.transformer_blocks[b].attention.W_key.bias = assign(gpt.transformer_blocks[b].attention.W_key.bias, k_b)
        gpt.transformer_blocks[b].attention.W_value.bias = assign(gpt.transformer_blocks[b].attention.W_value.bias, v_b)
        gpt.transformer_blocks[b].attention.out_projection.weight = assign(gpt.transformer_blocks[b].attention.out_projection.weight, params["blocks"][b]["attn"]["c_proj"]["w"].T)
        gpt.transformer_blocks[b].attention.out_projection.bias = assign(gpt.transformer_blocks[b].attention.out_projection.bias, params["blocks"][b]["attn"]["c_proj"]["b"])

        gpt.transformer_blocks[b].feed_forward.layers[0].weight = assign(gpt.transformer_blocks[b].feed_forward.layers[0].weight, params["blocks"][b]["mlp"]["c_fc"]["w"].T)
        gpt.transformer_blocks[b].feed_forward.layers[0].bias = assign(gpt.transformer_blocks[b].feed_forward.layers[0].bias, params["blocks"][b]["mlp"]["c_fc"]["b"])
        gpt.transformer_blocks[b].feed_forward.layers[2].weight = assign(gpt.transformer_blocks[b].feed_forward.layers[2].weight, params["blocks"][b]["mlp"]["c_proj"]["w"].T)
        gpt.transformer_blocks[b].feed_forward.layers[2].bias = assign(gpt.transformer_blocks[b].feed_forward.layers[2].bias,params["blocks"][b]["mlp"]["c_proj"]["b"])


        gpt.transformer_blocks[b].layer_norm1.scale = assign(gpt.transformer_blocks[b].layer_norm1.scale, params["blocks"][b]["ln_1"]["g"])
        gpt.transformer_blocks[b].layer_norm1.shift = assign(gpt.transformer_blocks[b].layer_norm1.shift, params["blocks"][b]["ln_1"]["b"])
        gpt.transformer_blocks[b].layer_norm2.scale = assign(gpt.transformer_blocks[b].layer_norm2.scale,params["blocks"][b]["ln_2"]["g"])
        gpt.transformer_blocks[b].layer_norm2.shift = assign(gpt.transformer_blocks[b].layer_norm2.shift,params["blocks"][b]["ln_2"]["b"])


    gpt.final_layer_norm.scale = assign(gpt.final_layer_norm.scale, params["g"])
    gpt.final_layer_norm.shift = assign(gpt.final_layer_norm.shift, params["b"])
    gpt.out_head.weight = assign(gpt.out_head.weight, params["wte"])

