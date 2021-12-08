import torch
import torch.nn as nn
from transformer import *
import torch.nn.init as init
import torch.autograd as autograd


class Discriminator(nn.Module):
    def __init__(self, encoder_decoder):
        super(Discriminator, self).__init__()
        self.encoder_decoder = encoder_decoder
    def forward(self, source, target, source_mask, target_mask):
        embeddings = self.encoder_decoder(source, target, source_mask, target_mask) # allow full attention for decoder by trg_mask=None
        logits = self.encoder_decoder.generator.proj(embeddings) # linear projection
        real_prob = torch.sigmoid(logits)[:,0,0] # use only the first time step
        return real_prob

def make_model(vocab_size, N=6, d_model=512, d_ff=2048, h=8, dropout=0.1, oracle_var=0.5):
    "Helper: Construct a model from hyperparameters."
    d_ff = d_model * 4

    c = copy.deepcopy
    attn = MultiHeadedAttention(h, d_model)
    ff = PositionwiseFeedForward(d_model, d_ff, dropout)
    position = PositionalEncoding(d_model, dropout)

    oracle = EncoderDecoder(
        Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout), N),
        Decoder(DecoderLayer(d_model, c(attn), c(attn), 
                             c(ff), dropout), N),
        nn.Sequential(Embeddings(d_model, vocab_size), c(position)),
        nn.Sequential(Embeddings(d_model, vocab_size), c(position)),
        Generator(d_model, vocab_size))

    generator = EncoderDecoder(
        Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout), N),
        Decoder(DecoderLayer(d_model, c(attn), c(attn), 
                             c(ff), dropout), N),
        nn.Sequential(Embeddings(d_model, vocab_size), c(position)),
        nn.Sequential(Embeddings(d_model, vocab_size), c(position)),
        Generator(d_model, vocab_size))

    discriminator = Discriminator(EncoderDecoder(
        Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout), N),
        Decoder(DecoderLayer(d_model, c(attn), c(attn), 
                             c(ff), dropout), N),
        nn.Sequential(Embeddings(d_model, vocab_size), c(position)),
        nn.Sequential(Embeddings(d_model, vocab_size), c(position)),
        Generator(d_model, 1)))

    for p in oracle.parameters():
        torch.nn.init.normal(p, 0, oracle_var)

    # This was important from their code. 
    # Initialize parameters with Glorot / fan_avg.
    params = list(generator.parameters()) + list(discriminator.parameters())
    for p in params:
        if p.dim() > 1:
            nn.init.xavier_uniform(p)

    # share embedding matrices
    oracle.tgt_embed[0].lut.weight = oracle.src_embed[0].lut.weight
    oracle.generator.proj.weight = oracle.src_embed[0].lut.weight

    generator.tgt_embed[0].lut.weight = generator.src_embed[0].lut.weight
    generator.generator.proj.weight = generator.src_embed[0].lut.weight

    discriminator.encoder_decoder.tgt_embed[0].lut.weight = discriminator.encoder_decoder.src_embed[0].lut.weight

    return oracle, generator, discriminator


class Source_Generator(nn.Module):

    def __init__(self, embedding_dim, hidden_dim, vocab_size, max_seq_len, gpu=False, oracle_init=False):
        super(Source_Generator, self).__init__()
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim
        self.max_seq_len = max_seq_len
        self.vocab_size = vocab_size
        self.gpu = gpu

        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.gru = nn.GRU(embedding_dim, hidden_dim)
        self.gru2out = nn.Linear(hidden_dim, vocab_size)

        # initialise oracle network with N(0,1)
        # otherwise variance of initialisation is very small => high NLL for data sampled from the same model
        if oracle_init:
            for p in self.parameters():
                init.normal(p, 0, 1)

    def init_hidden(self, batch_size=1):
        h = autograd.Variable(torch.zeros(1, batch_size, self.hidden_dim))

        if self.gpu:
            return h.cuda()
        else:
            return h

    def forward(self, inp, hidden):
        """
        Embeds input and applies GRU one token at a time (seq_len = 1)
        """
        # input dim                                             # batch_size
        emb = self.embeddings(inp)                              # batch_size x embedding_dim
        emb = emb.view(1, -1, self.embedding_dim)               # 1 x batch_size x embedding_dim
        out, hidden = self.gru(emb, hidden)                     # 1 x batch_size x hidden_dim (out)
        out = self.gru2out(out.view(-1, self.hidden_dim))       # batch_size x vocab_size
        out = F.log_softmax(out, dim=1)
        return out, hidden

    def sample(self, num_samples, start_letter=0):
        """
        Samples the network and returns num_samples samples of length max_seq_len.

        Outputs: samples, hidden
            - samples: num_samples x max_seq_length (a sampled sequence in each row)
        """

        samples = torch.zeros(num_samples, self.max_seq_len).type(torch.LongTensor)

        h = self.init_hidden(num_samples)
        inp = autograd.Variable(torch.LongTensor([start_letter]*num_samples))

        if self.gpu:
            samples = samples.cuda()
            inp = inp.cuda()

        for i in range(self.max_seq_len):
            out, h = self.forward(inp, h)               # out: num_samples x vocab_size
            out = torch.multinomial(torch.exp(out), 1)  # num_samples x 1 (sampling from each row)
            samples[:, i] = out.view(-1).data

            inp = out.view(-1)

        return samples
