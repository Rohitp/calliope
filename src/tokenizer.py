
# A sample inplementation of a tokeniser
# This is academic to understand behaviour
# Tiktoken is a standard BPE implementation that's easier and more convenient to use
import re
class Tokenizer():

    SPLIT_REGEX = r'([,.:;?_!"()\']|--|\s)'
    JOIN_REGEX = r'\s+([,.:;?!"()\'])'
    UNKNOWN = "<|unk|>"

    def __init__(self, lexicon):
        self.id_lookup = lexicon
        #Inverting the lexicon to get a word lookup
        # So we get {"x":1, "y":2} -> {1:"x", 2:"y"}
        self.word_lookup = {v: k for k, v in lexicon.items()}
    
    def encode(self, text):
        """
        Encode a string into a list of integers
        """
        tokens = re.split(self.SPLIT_REGEX, text)
        tokens = [token.strip() for token in tokens if token.strip()]
        encoded = [item if item in self.id_lookup else self.UNKNOWN for item in tokens]
        encoded = [self.id_lookup[token] for token in encoded]
        return encoded
    
    def decode(self, encoded):
        """
        Decode a list of integers into a string
        """
        decoded = [self.word_lookup[token] for token in encoded]
        decoded =  " ".join(decoded)
        # Since we split on whitespace, we need to remove any extra whitespace
        # Each token that is a punctuation becomes it's own token with spaces that need to be trimmed
        decoded = re.sub(self.JOIN_REGEX, r'\1', decoded)
        return decoded