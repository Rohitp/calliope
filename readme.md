#Calliope

[Calliope](https://en.wikipedia.org/wiki/Calliope) is a an LLM that's been written from scratch. While it can perform at roughly the level of [Phi-1.5](https://huggingface.co/microsoft/phi-1_5), it's meant to be a toy, for educational purposes only. Built with a lot of inspiration taken from [Alpaca](https://crfm.stanford.edu/2023/03/13/alpaca.html)

The other biggest influences while building this are

- Karpathy's extremely popular [video](https://www.youtube.com/watch?v=kCc8FmEb1nY) and his [nanoGPT](https://github.com/karpathy/nanoGPT)
- This [gist](https://gist.github.com/iamaziz/171170dce60d9cd07fab221507fd1d52) to get the bare basics and building blocks right
- Raschkas [book](https://www.amazon.co.uk/Build-Large-Language-Model-Scratch/dp/1633437167/)
- Just a lot of elbow grease and experimentation and sifting through arkane papers and youtube channels I can't even remember



The model is 163M and needs around 650MB to store the parameters. And it's been finetuned on the Alpaca [instruction dataset](https://raw.githubusercontent.com/tatsu-lab/stanford_alpaca/main/alpaca_data.json)

Obviously python based with torch, the main LLM modile is [Polymnia](https://en.wikipedia.org/wiki/Polyhymnia), with the transformer modules being called Optimus. Yes I was leaning into the greek mythology theme but couldn't resist the obvious transformer pun.

Trained on runpod with a bunch of A100s.

I will need to do more things to polish the code and the readme like - linking to the actual weights somewhere.
Linking to the charts that show how different hyperparameters affect training and validation loss and so on.

But for now, this is a version that works and is competent.

More References

- https://arxiv.org/abs/1706.03762 Multi head reference
- https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf -> LLM Reference

