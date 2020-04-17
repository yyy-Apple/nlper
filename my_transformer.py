import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import copy


# Embedding Layer
class Embedding(nn.Module):
    def __init__(self, vocab_size, d_model):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, d_model)
        self.d_model = d_model

    def forward(self, x):
        return self.emb(x) * math.sqrt(self.d_model)


# Position Encoding Layer
class PositionEncoding(nn.Module):
    def __init__(self, max_len, d_model, dropout=0.1):
        super().__init__()
        self.max_len = max_len
        self.d_model = d_model
        pos = torch.arange(0, max_len).unsqueeze(1)
        factor = 1 / (10000 ** (torch.arange(0, d_model, 2).float() / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(pos * factor)
        pe[:, 1::2] = torch.cos(pos * factor)
        pe = pe.unsqueeze(0)
        # pe should not be updated.
        self.register_buffer('pe', pe)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        # here x is batch x len x d_model
        return self.dropout(x + self.pe[:, :x.size(1), :])


# clone_layer function
def clone_layer(layer, n):
    return nn.ModuleList([copy.deepcopy(layer) for _ in range(n)])


# attention function
def attention(query, key, value, d_k, mask=None):
    # query: batch_size x h x query_len x d_k
    # key,value: batch_size x h x key_len x d_k
    probs = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    # here mask is just batch x h x query_len x key_len, it also could be query_len x key_len
    if mask is not None:
        probs = probs.masked_fill(mask == 0, -1e10)
    probs = F.softmax(probs, dim=-1)
    # return attn score for visualization purpose
    return torch.matmul(probs, value), probs


# MultiHeadAttention Layer
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, h, dropout=0.1):
        super().__init__()
        # we need to project the query, key, value, and the final result
        self.projs = clone_layer(nn.Linear(d_model, d_model), 4)
        self.d_model = d_model
        self.h = h
        self.d_k = self.d_model // h  # d_k = d_v
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)
        query = (self.projs[0](query)).view(batch_size, self.h, -1, self.d_k)
        key = (self.projs[1](key)).view(batch_size, self.h, -1, self.d_k)
        value = (self.projs[2](value)).view(batch_size, self.h, -1, self.d_k)
        out, _ = attention(query, key, value, self.d_k, mask)
        out = out.view(batch_size, -1, self.d_model)
        out = self.projs[-1](out)
        return self.dropout(out)


# FeedForward Layer
class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model)
        )
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        return self.dropout(self.ff(x))


# Connection Layer => connect a sublayer and residual connection and layernorm
# layer norm has only one parameter, which is the d_model in this case
class Connection(nn.Module):
    def __init__(self, d_model, dropout=0.1):
        super().__init__()
        self.layer_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, layer):
        # x: batch_size x len x d_model
        out = self.layer_norm(x + layer(x))
        return self.dropout(out)


# Encoder Layer: consists of two blocks, one contains multihead attention.
# The other contains feed forward network.
class EncoderLayer(nn.Module):
    def __init__(self, d_model, d_ff, h, dropout=0.1):
        super().__init__()
        self.att = MultiHeadAttention(d_model, h, dropout)
        self.ff = FeedForward(d_model, d_ff, dropout)

        self.att_block = Connection(d_model, dropout)
        self.ff_block = Connection(d_model, dropout)

    def forward(self, x, mask):
        out = self.att_block(x, lambda x: self.att(x, x, x, mask))
        out = self.ff_block(out, self.ff)
        return out


# Encoder: consists of 6 identical EncoderLayer
class Encoder(nn.Module):
    """ a stack of N = 6 identical Encoder Layers"""
    def __init__(self, encoder_layer, n):
        super().__init__()
        self.encoder_layers = clone_layer(encoder_layer, n)

    def forward(self, x, mask):
        for layer in self.encoder_layers:
            x = layer(x, mask)
        return x


# Decoder Layer
class DecoderLayer(nn.Module):
    def __init__(self, d_model, d_ff, h, dropout=0.1):
        super().__init__()
        self.self_att = MultiHeadAttention(d_model, h, dropout)
        self.enc_dec_att = MultiHeadAttention(d_model, h, dropout)
        self.ff = FeedForward(d_model, d_ff, dropout)

        self.blocks = clone_layer(Connection(d_model, dropout), 3)

    def forward(self, x, m, src_mask, tgt_mask):
        # first go through self attention
        out = self.blocks[0](x, lambda x: self.self_att(x, x, x, tgt_mask))
        # go through encoder-decoder attention, queries come from the previous decoder layer
        # and the memory keys and values come from the output of the encoder.
        out = self.blocks[1](out, lambda x: self.enc_dec_att(x, m, m, src_mask))
        # go through feed forward
        out = self.blocks[2](out, self.ff)
        return out


# Decoder
class Decoder(nn.Module):
    """ composed of a stack of N = 6 identical Decoder Layers"""
    def __init__(self, decoder_layer, n):
        super().__init__()
        self.decoder_layers = clone_layer(decoder_layer, n)

    def forward(self, x, m, src_mask, tgt_mask):
        for layer in self.decoder_layers:
            x = layer(x, m, src_mask, tgt_mask)
        return x


# Generator
class Generator(nn.Module):
    def __init__(self, d_model, tgt_vocab_size):
        super().__init__()
        self.proj = nn.Linear(d_model, tgt_vocab_size)

    def forward(self, x):
        # x: batch x len x d_model
        return F.log_softmax(self.proj(x), dim=-1)


# The final Transformer Model   (we need to add embedding and position encoding...)
class Transformer(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, max_len, d_model, d_ff, n, h, dropout=0.1):
        super().__init__()
        self.source_pe = nn.Sequential(
            Embedding(src_vocab_size, d_model),
            PositionEncoding(max_len, d_model, dropout)
        )
        self.target_pe = nn.Sequential(
            Embedding(tgt_vocab_size, d_model),
            PositionEncoding(max_len, d_model, dropout)
        )

        self.encoder = Encoder(EncoderLayer(d_model, d_ff, h, dropout), n)
        self.decoder = Decoder(DecoderLayer(d_model, d_ff, h, dropout), n)
        self.generator = Generator(d_model, tgt_vocab_size)

    def forward(self, src, tgt, src_mask, tgt_mask):
        m = self.encode(src, src_mask)
        return self.decode(tgt, m, src_mask, tgt_mask)

    def encode(self, src, src_mask):
        # src_mask is to mask out padding
        return self.encoder(self.source_pe(src), src_mask)

    def decode(self, tgt, m, src_mask, tgt_mask):
        # tgt_mask is to mask out padding and future words
        return self.decoder(self.target_pe(tgt), m, src_mask, tgt_mask)

    def generate(self, x):
        # x is batch_size x len x d_model. It's the output of decoder
        return self.generator(x)