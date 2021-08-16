from argparse import Namespace
import tokenizers
import torch
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence

import pytorch_lightning as pl

from model import Seq2SeqTransformer, VQSeq2SeqTransformer, create_mask
from data.dataset import Smiles2SmilesDataset
from data.tokenizer import load_tokenizer
from guacamol.utils.chemistry import canonicalize

class TransformerLightningModel(pl.LightningModule):
    def __init__(self, hparams):
        super(TransformerLightningModel, self).__init__()
        hparams = Namespace(**hparams) if isinstance(hparams, dict) else hparams
        self.save_hyperparameters(hparams)

        if hparams.continuous:
            self.model = Seq2SeqTransformer(
                num_encoder_layers=hparams.num_encoder_layers,
                num_decoder_layers=hparams.num_decoder_layers,
                emb_size=hparams.emb_size,
                nhead=hparams.nhead,
                src_vocab_size=400 + 4,
                tgt_vocab_size=400 + 4,
                dim_feedforward=hparams.dim_feedforward,
                dropout=hparams.dropout,
                )
        else:
            self.model = VQSeq2SeqTransformer(
                num_encoder_layers=hparams.num_encoder_layers,
                num_decoder_layers=hparams.num_decoder_layers,
                emb_size=hparams.emb_size,
                nhead=hparams.nhead,
                src_vocab_size=400 + 4,
                tgt_vocab_size=400 + 4,
                dim_feedforward=hparams.dim_feedforward,
                dropout=hparams.dropout,
                vq_codebook_size = hparams.vq_codebook_size,
                )

        self.dataset = Smiles2SmilesDataset(
            randomize_src=(not hparams.deterministic_src), 
            randomize_tgt=(not hparams.deterministic_tgt), 
            subsample_ratio=hparams.subsample_ratio
            )
        self.tokenizer = load_tokenizer()

    @staticmethod
    def add_args(parser):
        # Common - model
        parser.add_argument("--lr", type=float, default=1e-4)

        # Common - data
        parser.add_argument("--batch_size", type=int, default=128)
        parser.add_argument("--num_workers", type=int, default=8)

        parser.add_argument("--continuous", action="store_true")
        parser.add_argument("--num_encoder_layers", type=int, default=6)
        parser.add_argument("--num_decoder_layers", type=int, default=6)
        parser.add_argument("--emb_size", type=int, default=512)
        parser.add_argument("--nhead", type=int, default=8)
        parser.add_argument("--dim_feedforward", type=int, default=2048)
        parser.add_argument("--dropout", type=float, default=0.1)
        parser.add_argument("--vq_codebook_size", type=int, default=10)

        parser.add_argument("--deterministic_src", action="store_true")
        parser.add_argument("--deterministic_tgt", action="store_true")
        parser.add_argument("--subsample_ratio", type=float, default=1.0)
        
        return parser

    def train_dataloader(self):
        return DataLoader(
            self.dataset, batch_size=self.hparams.batch_size, shuffle=True, num_workers=self.hparams.num_workers,
        )

    def training_step(self, batched_data, batch_idx):
        loss, statistics = self.model.step(batched_data)
        
        self.log("train/loss/total", loss, on_step=True, logger=True)
        for key, val in statistics.items():
            self.log(f"train/{key}", val, on_step=True, logger=True)
            
        if (self.global_step + 1) % 100 == 0:
            src, tgt = batched_data
            src = src.transpose(1, 0)
            tgt = tgt.transpose(1, 0)
            tgt_input = tgt[:-1, :]    
            
            src_mask, _, src_key_padding_mask, _ = create_mask(src, tgt_input)
            smiles_list = self.tokenizer.decode_batch(src.transpose(1, 0).tolist())
            smiles_list = [smiles.replace(" ", "") for smiles in smiles_list]
            smiles_list = list(map(canonicalize, smiles_list))

            with torch.no_grad():
                self.model.eval()
                ys = self.model.decode_seq(src, src_mask, src_key_padding_mask)
                self.model.train()
            
            decoded_smiles_list = self.tokenizer.decode_batch(ys.transpose(1, 0).tolist())
            decoded_smiles_list = [smiles.replace(" ", "") for smiles in decoded_smiles_list]
            decoded_smiles_list = list(map(canonicalize, decoded_smiles_list))

            num_total = len(smiles_list)
            num_valid = len([0 for smiles in decoded_smiles_list if smiles is not None])
            num_correct = len(
                [0 for smiles, decoded_smiles in zip(smiles_list, decoded_smiles_list) if smiles == decoded_smiles]
            )

            self.log("train/stat/valid", float(num_valid) / num_total, on_step=True, logger=True)
            self.log("train/stat/correct", float(num_correct) / num_total, on_step=True, logger=True)
                
        return loss
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.hparams.lr, betas=(0.9, 0.98), eps=1e-9)
        return [optimizer]

if __name__ == "__main__":
    import argparse
    import torch
    import pytorch_lightning as pl
    from pytorch_lightning.callbacks import ModelCheckpoint
    from pytorch_lightning.loggers import NeptuneLogger

    parser = argparse.ArgumentParser()
    TransformerLightningModel.add_args(parser)
    parser.add_argument("--max_epochs", type=int, default=200)
    parser.add_argument("--gradient_clip_val", type=float, default=0.5)
    parser.add_argument("--checkpoint_path", type=str, default="../resource/checkpoint/default.pth")
    parser.add_argument("--tags", type=str, nargs="+", default=[])
    hparams = parser.parse_args()

    neptune_logger = NeptuneLogger(
        project_name="sungsahn0215/molrep", experiment_name="run_autoencoder", params=vars(hparams),
    )
    neptune_logger.append_tags(["autoencoder"] + hparams.tags)

    model = TransformerLightningModel(hparams)
    checkpoint_callback = ModelCheckpoint(monitor="train/loss/total")
    trainer = pl.Trainer(
        gpus=1,
        logger=neptune_logger,
        default_root_dir="../resource/log/",
        max_epochs=hparams.max_epochs,
        callbacks=[checkpoint_callback],
        gradient_clip_val=hparams.gradient_clip_val,
    )
    trainer.fit(model)
    model.load_from_checkpoint(checkpoint_callback.best_model_path)
    torch.save(model.state_dict(), hparams.checkpoint_path)
