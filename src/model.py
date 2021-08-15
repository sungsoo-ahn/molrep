from torch import Tensor
import torch
import torch.nn as nn
from torch.nn import Transformer
import math


# helper Module that adds positional encoding to the token embedding to introduce a notion of word order.
class PositionalEncoding(nn.Module):
    def __init__(self, emb_size, dropout, maxlen = 5000):
        super(PositionalEncoding, self).__init__()
        den = torch.exp(-torch.arange(0, emb_size, 2) * math.log(10000) / emb_size)
        pos = torch.arange(0, maxlen).reshape(maxlen, 1)
        pos_embedding = torch.zeros((maxlen, emb_size))
        pos_embedding[:, 0::2] = torch.sin(pos * den)
        pos_embedding[:, 1::2] = torch.cos(pos * den)
        pos_embedding = pos_embedding.unsqueeze(-2)

        self.dropout = nn.Dropout(dropout)
        self.register_buffer("pos_embedding", pos_embedding)

    def forward(self, token_embedding):
        return self.dropout(token_embedding + self.pos_embedding[: token_embedding.size(0), :])


# helper Module to convert tensor of input indices into corresponding tensor of token embeddings
class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size, emb_size):
        super(TokenEmbedding, self).__init__()
        self.embedding = nn.Embedding(vocab_size, emb_size)
        self.emb_size = emb_size

    def forward(self, tokens):
        return self.embedding(tokens.long()) * math.sqrt(self.emb_size)


# Seq2Seq Network
class Seq2SeqTransformer(nn.Module):
    def __init__(
        self,
        num_encoder_layers,
        num_decoder_layers,
        emb_size,
        nhead,
        src_vocab_size,
        tgt_vocab_size,
        dim_feedforward,
        dropout,
    ):
        super(Seq2SeqTransformer, self).__init__()
        self.transformer = Transformer(
            d_model=emb_size,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
        )
        self.generator = nn.Linear(emb_size, tgt_vocab_size)
        self.src_tok_emb = TokenEmbedding(src_vocab_size, emb_size)
        self.tgt_tok_emb = TokenEmbedding(tgt_vocab_size, emb_size)
        self.positional_encoding = PositionalEncoding(emb_size, dropout=dropout)

    def forward(
        self, src, trg, src_mask, tgt_mask, src_padding_mask, tgt_padding_mask, memory_key_padding_mask,
    ):
        src_emb = self.positional_encoding(self.src_tok_emb(src))
        tgt_emb = self.positional_encoding(self.tgt_tok_emb(trg))
        outs = self.transformer(
            src_emb, tgt_emb, src_mask, tgt_mask, None, src_padding_mask, tgt_padding_mask, memory_key_padding_mask
        )
        return self.generator(outs)

    def encode(self, src, src_mask, src_padding_mask):
        src_emb = self.positional_encoding(self.src_tok_emb(src))
        return self.transformer.encoder(src_emb, mask=src_mask, src_key_padding_mask=src_padding_mask)

    def decode(self, trg, memory, tgt_mask):
        tgt_emb = self.positional_encoding(self.tgt_tok_emb(trg))
        return self.transformer.decoder(tgt_emb, memory, tgt_mask=tgt_mask)
        
    def decode_seq(self, src, src_mask, src_padding_mask, pad_id=0, bos_id=2, eos_id=3, max_len=50):
        batch_size = src.size(1)
        memory = self.encode(src, src_mask, src_padding_mask)
        ys = torch.ones(1, batch_size).fill_(bos_id).type(torch.long).to(src.device)
        
        ended = torch.zeros(1, batch_size, dtype=torch.bool, device=src.device)
        for _ in range(max_len-1):
            tgt_mask = (generate_square_subsequent_mask(ys.size(0), src.device).type(torch.bool))
            out = self.decode(ys, memory, tgt_mask)
            out = out.transpose(0, 1)
            prob = self.generator(out[:, -1])
            next_word = torch.argmax(prob, dim=-1).unsqueeze(0)
            next_word[ended] = pad_id
            ys = torch.cat([ys, next_word], dim=0)
            ended = ended | (next_word == eos_id)
            
        return ys

def generate_square_subsequent_mask(sz, device):
    mask = (torch.triu(torch.ones((sz, sz), device=device)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float("-inf")).masked_fill(mask == 1, float(0.0))
    return mask


def create_mask(src, tgt):
    src_seq_len = src.shape[0]
    tgt_seq_len = tgt.shape[0]

    tgt_mask = generate_square_subsequent_mask(tgt_seq_len, device=tgt.device)
    src_mask = torch.zeros((src_seq_len, src_seq_len), device=src.device).type(torch.bool)

    src_padding_mask = (src == 0).transpose(0, 1)
    tgt_padding_mask = (tgt == 0).transpose(0, 1)
    return src_mask, tgt_mask, src_padding_mask, tgt_padding_mask