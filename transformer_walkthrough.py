# %% imports
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import copy
import torch.optim as optim


# %% Embedding Layer
class Embedding(nn.Module):
    def __init__(self, vocab_size, d_model):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, d_model)
        self.d_model = d_model

    def forward(self, x):
        return self.emb(x) * math.sqrt(self.d_model)


# %% Test Embedding Layer
# vocab_size: 10
# d_model: 32 (8 attention heads)
embedding = Embedding(10, 32)
# input: torch.tensor([[1, 3, 0, 2, 1], [2, 6, 9, 2, 1]], dtype=torch.long)  batch_size: 2 x len: 5
x1 = torch.tensor([[1, 3, 0, 2, 1], [2, 6, 9, 2, 1]], dtype=torch.long)
out = embedding(x1)  # should be shape of batch_size: 2 x len: 5 x d_model: 32
print(out)
print(out.size())
'''
Test Passed.
'''


# %% Position Encoding Layer
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


# %% Test Position Encoding Layer
# max_len: 10
# d_model: 32
embedding = Embedding(10, 32)
x1 = torch.tensor([[1, 3, 0, 2, 1], [2, 6, 9, 2, 1]], dtype=torch.long)
out1 = embedding(x1)
positionEncoding = PositionEncoding(10, 32)
out2 = positionEncoding(out1)
print(out2)
print(out2.size())  # should be the same as out1 which is batch_size: 2 x len: 5 x d_model: 32
'''
Test Passed.
'''


# %% clone_layer function
def clone_layer(layer, n):
    return nn.ModuleList([copy.deepcopy(layer) for _ in range(n)])


'''
Test Passed.
'''


# %% attention function
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


# %% Test attention function
# key: batch_size:2 x h:4 x len:5 x d_k:3
key = torch.tensor([[[[1, 2, 3], [2, 3, 1], [1, 3, 2], [5, 3, 5], [6, 7, 8]],
                     [[1, 2, 3], [2, 3, 1], [1, 3, 2], [5, 3, 5], [6, 7, 8]],
                     [[1, 2, 3], [2, 3, 1], [1, 3, 2], [5, 3, 5], [6, 7, 8]],
                     [[1, 2, 3], [2, 3, 1], [1, 3, 2], [5, 3, 5], [6, 7, 8]]],
                    [[[1, 2, 3], [2, 3, 1], [1, 3, 2], [5, 3, 5], [6, 7, 8]],
                     [[1, 2, 3], [2, 3, 1], [1, 3, 2], [5, 3, 5], [6, 7, 8]],
                     [[1, 2, 3], [2, 3, 1], [1, 3, 2], [5, 3, 5], [6, 7, 8]],
                     [[1, 2, 3], [2, 3, 1], [1, 3, 2], [5, 3, 5], [6, 7, 8]]]]).float()
query = copy.deepcopy(key)
value = copy.deepcopy(key)
mask = torch.tensor([[1, 1, 1, 0, 0], [1, 1, 0, 0, 0]]).unsqueeze(1).unsqueeze(1)
# mask = torch.tensor([[[1, 1, 1, 0, 0],
#                      [1, 1, 1, 0, 0],
#                      [1, 1, 1, 0, 0],
#                      [1, 1, 1, 0, 0],
#                      [1, 1, 1, 0, 0]],
#                      [[1, 1, 0, 0, 0],
#                       [1, 1, 0, 0, 0],
#                       [1, 1, 0, 0, 0],
#                       [1, 1, 0, 0, 0],
#                       [1, 1, 0, 0, 0]]
#                      ]).unsqueeze(1)
out_value, out_prob = attention(key, query, value, 3, mask)  # out_value shape should be 2 x 4 x 5 x 3
'''
Test Passed.
'''


# %% MultiHeadAttention Layer
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


# %% Test MultiHeadAttention Layer
# d_model:12 h:4
multiHeadAttention = MultiHeadAttention(12, 4)
# query: batch_size:2 x len:5 x d_model:12
query = torch.tensor([[[1, 1, 3, 21, 3, 2, 3, 1, 5, 7, 5, 8],
                       [1, 1, 3, 21, 3, 2, 3, 1, 5, 7, 5, 8],
                       [1, 1, 3, 21, 3, 2, 3, 1, 5, 7, 5, 8],
                       [1, 1, 3, 21, 3, 2, 3, 1, 5, 7, 5, 8],
                       [1, 1, 3, 21, 3, 2, 3, 1, 5, 7, 5, 8]],
                      [[1, 1, 3, 21, 3, 2, 3, 1, 5, 7, 5, 8],
                       [1, 1, 3, 21, 3, 2, 3, 1, 5, 7, 5, 8],
                       [1, 1, 3, 21, 3, 2, 3, 1, 5, 7, 5, 8],
                       [1, 1, 3, 21, 3, 2, 3, 1, 5, 7, 5, 8],
                       [1, 1, 3, 21, 3, 2, 3, 1, 5, 7, 5, 8]]]).float()
key = copy.deepcopy(query)
value = copy.deepcopy(query)
mask = torch.tensor([[1, 0, 0, 0, 0],
                     [1, 1, 0, 0, 0],
                     [1, 1, 1, 0, 0],
                     [1, 1, 1, 1, 0],
                     [1, 1, 1, 1, 1]])
multiHeadAttention_out = multiHeadAttention(query, key, value, mask)  # should be of size 2 x 5 x 12

'''
Test Passed.
'''


# %% FeedForward Layer
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


# %% Test FeedForward Layer
# inputTensor: batch_size:5 x len:10 x d_model:512
feedForward = FeedForward(512, 2048)
input_tensor = torch.ones(5, 10, 512)
out_tensor = feedForward(input_tensor)  # should be of size 5 x 10 x 512

'''
Test Passed.
'''


# %% Connection Layer => connect a sublayer and residual connection and layernorm
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


# %% Test Connection Layer
connection = Connection(512)
out_tensor2 = connection(input_tensor, feedForward)  # should be of size 5 x 10 x 512

'''
Test Passed.
'''


# %% Encoder Layer: consists of two blocks, one contains multihead attention.
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


# %% Test Encoder Layer
encoderLayer = EncoderLayer(512, 2048, 8)
in_tensor = torch.ones(5, 10, 512)  # batch_size:5 x len:10 x d_model:512
mask = torch.ones(10, 10).triu_(1)
mask = (mask == 0).long()
out_tensor = encoderLayer(in_tensor, mask)  # should be 5 x 10 x 512
'''
Test Passed.
'''


# %% Encoder: consists of 6 identical EncoderLayer
class Encoder(nn.Module):
    """ a stack of N = 6 identical Encoder Layers"""

    def __init__(self, encoder_layer, n):
        super().__init__()
        self.encoder_layers = clone_layer(encoder_layer, n)

    def forward(self, x, mask):
        for layer in self.encoder_layers:
            x = layer(x, mask)
        return x


# %% Test Encoder:
encoder = Encoder(EncoderLayer(512, 2048, 8), 6)
in_tensor = torch.ones(5, 10, 512)  # batch_size:5 x len:10 x d_model:512
mask = torch.ones(10, 10).triu_(1)
mask = (mask == 0).long()
out_tensor = encoderLayer(in_tensor, mask)  # should be 5 x 10 x 512
'''
Test Passed.
'''


# %% Decoder Layer
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


# %% Test Decoder Layer
decoderLayer = DecoderLayer(512, 2048, 8)
m = torch.rand(5, 10, 512)
in_tensor = torch.ones(5, 10, 512)  # batch_size:5 x len:10 x d_model:512
tgt_mask = torch.ones(10, 10).triu_(1)
tgt_mask = (mask == 0).long()
src_mask = None
out_tensor = decoderLayer(in_tensor, m, src_mask, tgt_mask)  # should be 5 x 10 x 512

'''
Test Passed.
'''


# %% Decoder
class Decoder(nn.Module):
    """ composed of a stack of N = 6 identical Decoder Layers"""

    def __init__(self, decoder_layer, n):
        super().__init__()
        self.decoder_layers = clone_layer(decoder_layer, n)

    def forward(self, x, m, src_mask, tgt_mask):
        for layer in self.decoder_layers:
            x = layer(x, m, src_mask, tgt_mask)
        return x


# %% Test Decoder
decoder = Decoder(DecoderLayer(512, 2048, 8), 6)
m = torch.rand(5, 10, 512)
in_tensor = torch.ones(5, 10, 512)  # batch_size:5 x len:10 x d_model:512
tgt_mask = torch.ones(10, 10).triu_(1)
tgt_mask = (mask == 0).long()
src_mask = None
out_tensor = decoderLayer(in_tensor, m, src_mask, tgt_mask)  # should be 5 x 10 x 512

'''
Test Passed.
'''


# %% Generator
class Generator(nn.Module):
    def __init__(self, d_model, tgt_vocab_size):
        super().__init__()
        self.proj = nn.Linear(d_model, tgt_vocab_size)

    def forward(self, x):
        # x: batch x len x d_model
        return F.log_softmax(self.proj(x), dim=-1)


# %% Test Generator
generator = Generator(512, 3000)
in_tensor = torch.ones(5, 10, 512)
out_tensor = generator(in_tensor)  # should be 5 x 10 x 3000

'''
Test Passed.
'''


# %% The final Transformer Model   (we need to add embedding and position encoding...)
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


# %% Test Transformer
transformer = Transformer(10, 10, 5000, 512, 2048, 6, 8)
for p in transformer.parameters():
    if p.dim() > 1:
        nn.init.xavier_uniform_(p)
# pad the input
src = torch.tensor([[1, 2, 0, 0], [1, 1, 1, 0]])
tgt = torch.tensor([[1, 4, 5, 2, 2], [1, 2, 4, 7, 8]])
# mask can be 4 dim like this
src_mask = (src != 0).unsqueeze(1).unsqueeze(1)
# or keep the last 2 dimension
tgt_mask = torch.tensor([[1, 0, 0, 0, 0],
                         [1, 1, 0, 0, 0],
                         [1, 1, 1, 0, 0],
                         [1, 1, 1, 1, 0],
                         [1, 1, 1, 1, 1]])
out_tensor = transformer(src, tgt, src_mask, tgt_mask)

'''
Test Passed.
'''


# %% Basic Training for a simple copy task
# When computing loss, we need to mask out pad on both the computed distribution
# and true distribution
def generate_data(n_batches, batch_size, vocab_size):
    # param: total number of batches, batchsize, vocabulary size
    # Here is how to generate random integer tensor in pytorch:
    # t = torch.tensor([whatever]), and then use t.random_(from, to) will convert each element to [from, to - 1] integer
    for i in range(n_batches):
        # Here we consider the easy case where no padding is in each batch, and 1 is the <BOS> symbol
        # default tensor does not require grad
        # convert the first tokens to <BOS> in the batch
        data = torch.zeros(batch_size, 10).random_(1, vocab_size).long()
        data[:, 0] = 1
        yield Batch(data, data, padding_idx=0)


class Batch:
    def __init__(self, src, tgt=None, padding_idx=0):
        self.src = src
        self.tgt = tgt
        self.src_mask = (src != padding_idx).long().unsqueeze(1).unsqueeze(1)
        if tgt is not None:
            self.tgt_x = tgt[:, : -1]
            self.tgt_y = tgt[:, 1:]
            # mask out the padding, for future loss compute use
            self.tgt_pad_mask = (self.tgt_y != padding_idx).long().unsqueeze(1).unsqueeze(1)
            # prevent from attending to future words
            self.tgt_pos_mask = self.make_pos_mask(self.tgt_y.size(-1))
            self.tgt_mask = self.tgt_pad_mask * self.tgt_pos_mask

            # mask out padding for loss computation
            self.loss_mask = (self.tgt_y == padding_idx).unsqueeze(-1)

    @staticmethod
    def make_pos_mask(len):
        mask = torch.ones(len, len)
        mask = (mask.triu(1) == 0).long()
        return mask


# %% Test generate_data and Batch
src = torch.tensor([[1, 2, 3, 0], [1, 6, 0, 0]])
tgt = copy.deepcopy(src)
# tgt_x = torch.tensor([[1, 2, 3], [1, 6, 0]])
# tgt_y = torch.tensor([[2, 3, 0],[6, 0, 0]])
batch = Batch(src, tgt, 0)
# batch.src_mask should be tensor([[[[1, 1, 1, 0]]], [[[1, 1, 0, 0]]]])
# batch.tgt_pad_mask should be tensor([[[[1, 1, 0]]], [[[1, 0, 0]]]])
# batch.tgt_pos_mask should be tensor([[1, 0, 0], [1, 1, 0], [1, 1, 1]])
# batch.tgt_mask should be
# tensor([[[[1, 0, 0],
#           [1, 1, 0],
#           [1, 1, 0]]],
#         [[[1, 0, 0],
#           [1, 0, 0],
#           [1, 0, 0]]]])
for i, batch in enumerate(generate_data(1, 10, 20)):
    print(batch.src)
    print(batch.tgt_x)
    print(batch.tgt_y)

'''
Test Passed.
'''


# %% Loss Function
def compute_loss(pred, tgt_y, loss_mask, smooth=0.1):
    # pred is batch x len x vocab_size
    tgt = torch.empty_like(pred)
    tgt.fill_(smooth / (tgt.size(-1) - 1))
    tgt.scatter_(-1, tgt_y.unsqueeze(-1), (1 - smooth))
    # mask out padding
    tgt.masked_fill_(loss_mask, 0)
    pred = pred.masked_fill(loss_mask, 0)
    loss_func = nn.KLDivLoss(reduction='mean')
    loss = loss_func(pred, tgt)
    return loss


# %% Test Loss Function
pred = torch.tensor([[[0.3, 0.2, 0.5], [0.3, 0.2, 0.5], [0.3, 0.2, 0.5]],
                     [[0.1, 0.8, 0.1], [0.1, 0.8, 0.1], [0.1, 0.8, 0.1]]])
tgt_y = torch.tensor([[1, 2, 0], [1, 1, 2]], dtype=torch.long)
loss_mask = (tgt_y == 0).unsqueeze(-1)
compute_loss(pred, tgt_y, loss_mask, 0.1)

'''
Test Passed
'''


# %% Optimizer
class Optimizer:
    def __init__(self, model, d_model, warmup_steps):
        self.optimizer = optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.98), eps=1e-9)
        self.d_model = d_model
        self.warmup_steps = warmup_steps
        self.steps = 0
        self.lr = None

    def step(self):
        self.steps += 1
        lr = self.rate()
        self.lr = lr
        for p in self.optimizer.param_groups:
            p['lr'] = lr
        self.optimizer.step()

    def rate(self):
        return self.d_model ** (-0.5) * min(self.steps ** (-0.5), self.steps * self.warmup_steps ** (-1.5))

    def zero_grad(self):
        self.optimizer.zero_grad()


# %% Test Optimizer
transformer = Transformer(10, 10, 5000, 512, 2048, 6, 8)
for p in transformer.parameters():
    if p.dim() > 1:
        nn.init.xavier_uniform_(p)
optimizer = Optimizer(transformer, 512, 4000)
optimizer.step()

'''
Test Passed
'''

# %% Begin Training
transformer = Transformer(10, 10, 5000, 512, 2048, 2, 8)
for p in transformer.parameters():
    if p.dim() > 1:
        nn.init.xavier_uniform_(p)
optimizer = Optimizer(transformer, 512, 400)

transformer.train()
for epoch in range(50):
    running_loss = 0.0
    for i, batch in enumerate(generate_data(30, 30, 10)):
        src = batch.src
        src_mask = batch.src_mask
        tgt = batch.tgt
        if tgt is not None:
            optimizer.zero_grad()
            tgt_x = batch.tgt_x
            tgt_y = batch.tgt_y
            tgt_mask = batch.tgt_mask
            memory = transformer.encode(src, src_mask)
            out_decoder = transformer.decode(tgt_x, memory, src_mask, tgt_mask)
            pred = transformer.generate(out_decoder)
            print((tgt_y == pred.argmax(dim=-1)).sum().item())
            print((tgt_y != 0).sum().item())
            loss_mask = batch.loss_mask
            loss = compute_loss(pred, tgt_y, loss_mask)
            running_loss = loss.item()
            loss.backward()
            optimizer.step()
    print("On epoch %d, the current loss is %f" % (epoch, running_loss))

'''
Test Passed.
'''
# %% Inference
transformer.eval()


def inference(model, src, src_mask, max_len):
    memory = model.encode(src, src_mask)
    generated_tokens = [1]  # start symbol
    for i in range(max_len - 1):
        gen_len = len(generated_tokens)
        tgt_mask = torch.ones(gen_len, gen_len)
        tgt_mask = (tgt_mask.triu(1) == 0).long()
        out = model.decode(torch.tensor(generated_tokens).unsqueeze(0), memory, src_mask, tgt_mask)
        next_token = model.generate(out[:, -1]).argmax(dim=-1).squeeze().item()
        generated_tokens.append(next_token)
    print(generated_tokens)
    return generated_tokens


inference(transformer, torch.tensor([1, 2, 1, 2, 2, 2, 2, 2, 2]).unsqueeze(0),
          None, 9)

'''
Test Passed.
'''
