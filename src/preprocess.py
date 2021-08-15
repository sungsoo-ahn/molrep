from pathlib import Path
import random
import torch

from tokenizers import Tokenizer
from tokenizers import pre_tokenizers
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.processors import TemplateProcessing

if __name__ == "__main__":
    DATA_DIR = "../resource/data"
    src_file = Path(f"{DATA_DIR}/smiles_list.txt")
    smiles_list = src_file.read_text(encoding="utf-8").splitlines()
    
    tokenizer = Tokenizer(BPE())
    tokenizer.pad_token = "[PAD]"
    tokenizer.bos_token = "[BOS]"
    tokenizer.eos_token = "[EOS]"
    tokenizer.pre_tokenizer = pre_tokenizers.Split(
        "(\[|\]|Br?|Cl?|Si?|Se?|se?|@@?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\\\|\/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])",
        "isolated",
    )
    trainer = BpeTrainer(vocab_size=400, special_tokens=["[PAD]", "[MASK]", "[BOS]", "[EOS]"])
    tokenizer.train_from_iterator(iter(smiles_list), trainer)
    tokenizer.post_processor = TemplateProcessing(
        single="[BOS] $A [EOS]",
        special_tokens=[("[BOS]", tokenizer.token_to_id("[BOS]")), ("[EOS]", tokenizer.token_to_id("[EOS]")),],
    )
    tokenizer.save(f"{DATA_DIR}/tokenizer.json")
    