import torch
import torch.nn.functional as F
from torch import nn
from torchtext.vocab import build_vocab_from_iterator
from torch.autograd import Variable
from torch.utils.data import TensorDataset, DataLoader

import sacrebleu

import numpy as np
from tqdm import tqdm
import math, copy, time



device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

PAD_TOKEN = '<pad>'
UNK_TOKEN = '<unk>'
BOS_TOKEN = '<bos>'
EOS_TOKEN = '<eos>'
SPECIAL_TOKENS = [PAD_TOKEN, UNK_TOKEN, BOS_TOKEN, EOS_TOKEN]
MAX_SEQUENCE_LENGTH = 83



def subsequent_mask(size: int):
    attn_shape = (1, size, size)
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    return torch.from_numpy(subsequent_mask) == 0


def dataset_iterator(path):
    with open(path) as texts:
        for text in texts:
            yield text.split()


def build_vocab(path, min_freq):
    return build_vocab_from_iterator(
        dataset_iterator(path),
        specials=SPECIAL_TOKENS, min_freq=min_freq,
    )


def get_train_val_dataloaders(path, en_vocab, de_vocab, batch_size) -> list[DataLoader]:
    train_en_tokens = []
    for text in dataset_iterator(f'{path}/data/train.de-en.en'):
        tokens = [2] + [en_vocab[word] if word in en_vocab else en_vocab[UNK_TOKEN] for word in text] + [3]
        train_en_tokens += [tokens]

    train_de_tokens = []
    for text in dataset_iterator(f'{path}/data/train.de-en.de'):
        tokens = [2] + [de_vocab[word] if word in de_vocab else de_vocab[UNK_TOKEN] for word in text] + [3]
        train_de_tokens += [tokens]

    test_en_tokens = []
    for text in dataset_iterator(f'{path}/data/val.de-en.en'):
        tokens = [2] + [en_vocab[word] if word in en_vocab else en_vocab[UNK_TOKEN] for word in text] + [3]
        test_en_tokens += [tokens]

    test_de_tokens = []
    for text in dataset_iterator(f'{path}/data/val.de-en.de'):
        tokens = [2] + [de_vocab[word] if word in de_vocab else de_vocab[UNK_TOKEN] for word in text] + [3]
        test_de_tokens += [tokens]


    max_length = MAX_SEQUENCE_LENGTH
    tokenized_en_train = torch.full((len(train_en_tokens), max_length), en_vocab[PAD_TOKEN], dtype=torch.int32)
    for i, tokens in enumerate(train_en_tokens):
        length = min(max_length, len(tokens))
        tokenized_en_train[i, :length] = torch.tensor(tokens[:length])

    tokenized_de_train = torch.full((len(train_de_tokens), max_length), de_vocab[PAD_TOKEN], dtype=torch.int32)
    for i, tokens in enumerate(train_de_tokens):
        length = min(max_length, len(tokens))
        tokenized_de_train[i, :length] = torch.tensor(tokens[:length])

    tokenized_en_test = torch.full((len(test_en_tokens), max_length), en_vocab[PAD_TOKEN], dtype=torch.int32)
    for i, tokens in enumerate(test_en_tokens):
        length = min(max_length, len(tokens))
        tokenized_en_test[i, :length] = torch.tensor(tokens[:length])

    tokenized_de_test = torch.full((len(test_de_tokens), max_length), de_vocab[PAD_TOKEN], dtype=torch.int32)
    for i, tokens in enumerate(test_de_tokens):
        length = min(max_length, len(tokens))
        tokenized_de_test[i, :length] = torch.tensor(tokens[:length])

    train_dataset = TensorDataset(tokenized_de_train, tokenized_en_train)
    test_dataset = TensorDataset(tokenized_de_test, tokenized_en_test)

    train_loader = DataLoader(train_dataset, batch_size, shuffle=True)
    val_loader = DataLoader(test_dataset, batch_size, shuffle=False)

    return train_loader, val_loader


def clones(module: nn.Module, N: int) -> nn.ModuleList:
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


def Attention(query, key, value, mask=None, dropout=None):
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = F.softmax(scores, dim = -1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value)


class LayerNorm(nn.Module):
    def __init__(self, size: int, eps: float =1e-6) -> None:
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(size))
        self.b_2 = nn.Parameter(torch.zeros(size))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2
    

class SublayerConnection(nn.Module):
    def __init__(self, size: int, dropout: float):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))


class Batch:
    def __init__(self, src, trg=None, pad=0):
        self.src = src
        self.src_mask = (src != pad).unsqueeze(-2)
        if trg is not None:
            self.trg = trg[:, :-1]
            self.trg_y = trg[:, 1:]
            self.trg_mask = self.make_std_mask(self.trg, pad)
            self.ntokens = (self.trg_y != pad).data.sum()
    
    def make_std_mask(self, tgt, pad):
        tgt_mask = (tgt != pad).unsqueeze(-2)
        tgt_mask = tgt_mask & Variable(
            subsequent_mask(tgt.size(-1)).type_as(tgt_mask.data))
        return tgt_mask
    

class Embeddings(nn.Module):
    def __init__(self, d_model: int, vocab_size: int):
        super(Embeddings, self).__init__()
        self.emb = nn.Embedding(vocab_size, d_model)
        self.d_model = d_model

    def forward(self, x):
        return self.emb(x) * math.sqrt(self.d_model)
    

class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))
    

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        x = x + Variable(self.pe[:, :x.size(1)], 
                         requires_grad=False)
        return self.dropout(x)


class MultiHeadedAttention(nn.Module):
    def __init__(self, h: int, d_model: int, dropout: float = 0.1):
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.dropout = nn.Dropout(p=dropout)
        
    def forward(self, query, key, value, mask=None):
        if mask is not None:
            mask = mask.unsqueeze(1)
        batch_size = query.size(0)
        
        query, key, value = [l(x).view(batch_size, -1, self.h, self.d_k).transpose(1, 2) for l, x in zip(self.linears, (query, key, value))]
        x = Attention(query, key, value, mask=mask, dropout=self.dropout)
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.h * self.d_k)
        return self.linears[-1](x)
    

class EncoderLayer(nn.Module):
    def __init__(self, size: int, self_attn, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.size = size

    def forward(self, x, mask):
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        return self.sublayer[1](x, self.feed_forward)
    

class Encoder(nn.Module):
    def __init__(self, layer, N):
        super(Encoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)
        
    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)
    

class DecoderLayer(nn.Module):
    def __init__(self, size, self_attn, src_attn, feed_forward, dropout):
        super(DecoderLayer, self).__init__()
        self.size = size
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 3)
 
    def forward(self, x, memory, src_mask, tgt_mask):
        m = memory
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask))
        x = self.sublayer[1](x, lambda x: self.src_attn(x, m, m, src_mask))
        return self.sublayer[2](x, self.feed_forward)


class Decoder(nn.Module):
    def __init__(self, layer, N):
        super(Decoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)
        
    def forward(self, x, memory, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
        return self.norm(x)
    

class Transformer(nn.Module):
    def __init__(self, encoder, decoder, src_embed, tgt_embed, generator):
        super(Transformer, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.generator = generator
        
    def forward(self, src, tgt, src_mask, tgt_mask):
        return self.decode(self.encode(src, src_mask), src_mask,
                            tgt, tgt_mask)
    
    def encode(self, src, src_mask):
        return self.encoder(self.src_embed(src), src_mask)
    
    def decode(self, memory, src_mask, tgt, tgt_mask):
        return self.decoder(self.tgt_embed(tgt), memory, src_mask, tgt_mask)
    

class Generator(nn.Module):
    def __init__(self, d_model, vocab):
        super(Generator, self).__init__()
        self.linear = nn.Linear(d_model, vocab)

    def forward(self, x):
        return self.linear(x)
    

def make_transformer(src_vocab, tgt_vocab, N, 
               d_model, d_ff, h, dropout=0.1):
    c = copy.deepcopy
    attn = MultiHeadedAttention(h, d_model)
    ff = PositionwiseFeedForward(d_model, d_ff, dropout)
    position = PositionalEncoding(d_model, dropout)
    model = Transformer(
        Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout), N),
        Decoder(DecoderLayer(d_model, c(attn), c(attn), 
                             c(ff), dropout), N),
        nn.Sequential(Embeddings(d_model, src_vocab), c(position)),
        nn.Sequential(Embeddings(d_model, tgt_vocab), c(position)),
        Generator(d_model, tgt_vocab))
    
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    return model


class WarmupOptimizer:
    def __init__(self, model_size, factor, warmup, optimizer):
        self.optimizer = optimizer
        self._step = 0
        self.warmup = warmup
        self.factor = factor
        self.model_size = model_size
        self._rate = 0
        
    def step(self):
        self._step += 1
        rate = self.rate()
        for p in self.optimizer.param_groups:
            p['lr'] = rate
        self._rate = rate
        self.optimizer.step()
        
    def rate(self, step = None):
        if step is None:
            step = self._step
        return self.factor * (self.model_size ** (-0.5) * min(step ** (-0.5), step * self.warmup ** (-1.5)))
    

class LossBLEUCompute:
    def __init__(self, generator, criterion, en_vocab_reversed, calc_bleu, opt=None):
        self.generator = generator
        self.criterion = criterion
        self.en_vocab_reversed = en_vocab_reversed
        self.calc_bleu = calc_bleu
        self.opt = opt
    
        
    def __call__(self, x, y):
        x = self.generator(x)
        bleu = 0
        if self.calc_bleu:
            next_tokens = torch.argmax(x, dim=2)
            predicted_sentences = [' '.join([self.en_vocab_reversed[word] for word in next_sentence]) + '\n' for next_sentence in next_tokens]
            reference_sentences = [' '.join([self.en_vocab_reversed[word] for word in next_sentence]) + '\n' for next_sentence in y]
            bleu = sacrebleu.corpus_bleu(reference_sentences, [predicted_sentences]).score
        
        loss = self.criterion(x.transpose(1, 2), 
                              y.long())

        loss.backward()
        
        if self.opt is not None:
            self.opt.step()
            self.opt.optimizer.zero_grad()

        return loss.item(), bleu
    

def run_epoch(data_loader, model, loss_bleu_compute):
    start = time.time()
    total_tokens = 1
    total_loss = 0
    total_bleu = 0
    tokens = 0
    for i, src_trg in tqdm(enumerate(data_loader)):
        src, trg = src_trg
        batch = Batch(src, trg, 0)
        batch.src = batch.src.to(device)
        batch.trg = batch.trg.to(device)
        batch.src_mask = batch.src_mask.to(device)
        batch.trg_mask = batch.trg_mask.to(device)
        batch.trg_y = batch.trg_y.to(device)
        batch.ntokens = batch.ntokens.to(device)
        
        out = model.forward(batch.src, batch.trg, 
                            batch.src_mask, batch.trg_mask)
        loss, bleu = loss_bleu_compute(out, batch.trg_y)
        total_loss += loss
        total_bleu += bleu
        total_tokens += batch.ntokens
        tokens += batch.ntokens
        if i % 100 == 1:
            elapsed = time.time() - start
            print("Step: %d Loss: %f Tokens per Sec: %f" % (i, loss / batch.ntokens, tokens / elapsed))
            start = time.time()
            tokens = 0
    return total_loss / total_tokens, total_bleu / total_tokens


def train_loop(num_epochs, model, train_loader, test_loader, train_loss_comp, test_loss_comp, config):
    train_loss, val_loss = [], []
    train_bleu, val_bleu = [], []
    for epoch in range(num_epochs):
        model.train()
        loss, bleu = run_epoch(train_loader, model, train_loss_comp)
        train_loss.append(loss)
        train_bleu.append(bleu)
        model.eval()
        loss, bleu = run_epoch(test_loader, model, test_loss_comp)
        val_loss.append(loss)
        val_bleu.append(bleu)
        print(f'train loss after {epoch}th epoch: {train_loss[-1]}')
        print(f'val loss after {epoch}th epoch: {val_loss[-1]}')


def inference_loop(model, tokenized_src, en_vocab, max_length=90):
    en_reverse_vocab = en_vocab.get_itos()

    model.eval()
    trg_tokens = [en_vocab["<bos>"]]
    
    for _ in range(max_length):
        src = torch.tensor([tokenized_src]).to(device)
        trg = torch.tensor([trg_tokens]).to(device)
        with torch.no_grad():
            output = model.forward(src, trg, None, None)
        prob_distribution = model.generator(output[:, -1])
        next_token = torch.argmax(prob_distribution, dim=1).item()
        if next_token == en_vocab["<eos>"]:
            break
        trg_tokens.append(next_token)
    
    translated_sentence = [en_reverse_vocab[token] for token in trg_tokens][1:]
    
    return translated_sentence


def main():
    config = {
        'model': {
            'num_layers': 2,
            'embedding_dim': 128,
            'feedforward_dim': 256,
            'num_heads': 4,
            'dropout': 0.1
        },
        'batch_size': 1024,
        'optimizer': {
            'factor': 1,
            'warmup': 400,
            'lr': 0.0001,
            'beta1': 0.9,
            'beta2': 0.98
        },
        'epochs': 5,
        'checkpoint': {
            'dir': 'checkpoints',
            'step': 1
        },
        'outputfile': 'output',
        'path': '../',
        'run_name': 'main',
        'min_freq': 10
    }


    path = config['path']

    en_vocab = build_vocab(f'{path}/data/train.de-en.en', config['min_freq'])
    de_vocab = build_vocab(f'{path}/data/train.de-en.de', config['min_freq'])
    train_loader, val_loader = get_train_val_dataloaders(path=path, en_vocab=en_vocab, de_vocab=de_vocab, batch_size=config['batch_size'])
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    model = make_transformer(len(de_vocab), len(en_vocab),
                            N=config['model']['num_layers'],
                            d_model=config['model']['embedding_dim'],
                            d_ff=config['model']['feedforward_dim'], 
                            h=config['model']['num_heads']).to(device)
    optimizer = WarmupOptimizer(model.src_embed[0].d_model, config['optimizer']['factor'], 
                                config['optimizer']['warmup'],
                                torch.optim.Adam(model.parameters(), lr=config['optimizer']['lr'], 
                                                betas=(config['optimizer']['beta1'], 
                                                config['optimizer']['beta2']), eps=1e-09))
    
    train_loop(num_epochs=config['epochs'], model=model, train_loader=train_loader,
               test_loader=val_loader,
               train_loss_comp=LossBLEUCompute(model.generator, criterion, en_vocab.get_itos(), calc_bleu=False, opt=optimizer),
               test_loss_comp=LossBLEUCompute(model.generator, criterion, en_vocab.get_itos(), calc_bleu=False, opt=None), config=config)

    with open(config['outputfile'], 'w') as ans_file:
        for text in dataset_iterator(f'{path}/data/test1.de-en.de'):
            tokens = [2] + [de_vocab[word] if word in de_vocab else de_vocab['<unk>'] for word in text] + [3]
            ans_file.write(' '.join(inference_loop(model=model, tokenized_src=tokens, en_vocab=en_vocab)) + '\n')


if __name__ == '__main__':
    main()
