import re
from tokenizer import Tokenizer

with open("../text/alice.txt", "r") as f:
    calliope = f.read()



test = "“Well!” thought Alice to herself, “after such a fall as this, I shall think nothing of tumbling down stairs!"

# Splitting on whitespace and punctuation
# Copied from deepseek. I've forgotten regular expressions. Seems to work good enough
pattern = r'([,.:;?_!"()\']|--|\s)'
tokens = re.split(pattern, calliope)

text = [token.strip() for token in tokens if token.strip()]
vocabulary = sorted(set(text))


# print(len(vocabulary))


# A set of all unique lexemes in the text
lexicon = {word: i for i, word in enumerate(vocabulary)}
to_int = {i : word for i, word in lexicon.items()}
for i, word in enumerate(to_int.items()):
    if i < 10:
        # print(f"{i}: {word}")
        pass
    else:
        break


tokenizer = Tokenizer(lexicon)
print(tokenizer.encode(test))
print(tokenizer.decode(tokenizer.encode(test)))