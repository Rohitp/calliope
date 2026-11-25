# Calliope

Calliope is a 163M-parameter language model I built from scratch in PyTorch to learn how transformers work. I trained it on A100s and fine-tuned it on the [Alpaca instruction dataset](https://github.com/tatsu-lab/stanford_alpaca). The parameters take up about 650 MB.

## Start with the code

- [Polymnia](src/modules/polymnia.py) puts the model together.
- [Optimus](src/modules/optimus.py) implements the transformer block. I was going for Greek mythology, but I couldn't pass up the transformer pun.
- [trainer.py](src/trainer.py) and the [training helpers](src/train) are the place to look for the training code.

## What I learned from

- Karpathy's extremely popular [video](https://www.youtube.com/watch?v=kCc8FmEb1nY) and his [nanoGPT](https://github.com/karpathy/nanoGPT)
- This [gist](https://gist.github.com/iamaziz/171170dce60d9cd07fab221507fd1d52) to get the bare basics and building blocks right
- Raschkas [book](https://www.amazon.co.uk/Build-Large-Language-Model-Scratch/dp/1633437167/)
- Just a lot of elbow grease and experimentation and sifting through arcane papers and youtube channels I can't even remember

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- [Language Models are Unsupervised Multitask Learners](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)
