trying to write my own GPT2 equivalent LLM from scratch. 

-> https://arxiv.org/abs/1706.03762 Multi head reference
-> https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf -> LLM Reference



So this is a 163M model, which is slightly higher than the 124M GPT2 model. 
Because there is no weight tying done, the input and output weight embeddings are different.

The total memory needed = 621.83MB, assuming each parameter needs 4 bytes