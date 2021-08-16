from pathlib import Path
import torch
from torch.utils.data import Dataset
from data.tokenizer import load_tokenizer

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from rdkit import Chem

def randomize_smiles(smiles):
    mol = Chem.MolFromSmiles(smiles)
    smiles = Chem.MolToSmiles(mol, canonical=False, doRandom=True, isomericSmiles=False)
    return smiles

class Smiles2SmilesDataset(Dataset):
    def __init__(self, randomize_src=False, randomize_tgt=False, randomize_sync=False, subsample_ratio=1.0, max_len=50):
        self.smiles_list = Path("../resource/data/smiles_list.txt").read_text(encoding="utf-8").splitlines()
        if subsample_ratio < 1.0:
            subsample_len = int(subsample_ratio * len(self.smiles_list))
            self.smiles_list = self.smiles_list[:subsample_len]
            
        self.tokenizer = load_tokenizer()
        self.max_len = max_len
        self.randomize_src = randomize_src
        self.randomize_tgt = randomize_tgt
        self.randomize_sync = randomize_sync

    def __len__(self):
        return len(self.smiles_list)

    def __getitem__(self, idx):
        smiles0 = smiles1 = self.smiles_list[idx]
        if self.randomize_src:
            smiles0 = randomize_smiles(smiles0)

        if self.randomize_tgt:    
            smiles1 = randomize_smiles(smiles1)

        if self.randomize_sync:
            smiles0 = smiles1 = randomize_smiles(smiles0)

        x0 = self.tensorize(smiles0)
        x1 = self.tensorize(smiles1)
        
        return torch.tensor(x0), torch.tensor(x1)
    
    def tensorize(self, smiles):
        x = self.tokenizer.encode(smiles).ids
        x = x + [self.tokenizer.token_to_id("[PAD]")] * (self.max_len - len(x))
        
        try:
            assert smiles == self.tokenizer.decode(x).replace(" ", "")
        except:
            print(smiles)
            print(self.tokenizer.decode(x).replace(" ", ""))
            assert False

        return x


if __name__ == "__main__":
    tokenizer = load_tokenizer()
    dataset = Smiles2SmilesDataset()
    print(tokenizer.decode(dataset[0].tolist(), clean_up_tokenization_spaces=True))
