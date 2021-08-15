from tokenizers import Tokenizer

DATA_DIR = "../resource/data"

def load_tokenizer():
    tokenizer = Tokenizer.from_file(f"{DATA_DIR}/tokenizer.json")
    return tokenizer