import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import random


def subsequent_mask(size):
    "Mask out subsequent positions."
    attn_shape = (1, size, size)
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    return torch.from_numpy(subsequent_mask) == 0

class Batch:
    "Object for holding a batch of data with mask during training."
    def __init__(self, src, trg=None, pad=2):
        self.src = src
        self.src_mask = (src != pad).unsqueeze(-2)
        if trg is not None:
            self.trg = trg[:, :-1]
            self.trg_y = trg[:, 1:]
            self.trg_mask = \
                self.make_std_mask(self.trg, pad)
            self.ntokens = (self.trg_y != pad).data.sum()
    
    def make_cuda(self,device):
        self.src = self.src.cuda(device)
        self.trg = self.trg.cuda(device)
        self.trg_y = self.trg_y.cuda(device)
        self.src_mask = self.src_mask.cuda(device)
        self.trg_mask = self.trg_mask.cuda(device)

    @staticmethod
    def make_std_mask(tgt, pad):
        "Create a mask to hide padding and future words."
        tgt_mask = (tgt != pad).unsqueeze(-2)
        tgt_mask = tgt_mask & Variable(
            subsequent_mask(tgt.size(-1)).type_as(tgt_mask.data))
        return tgt_mask

class NoamOpt:
    "Optim wrapper that implements rate."
    def __init__(self, model_size, factor, warmup, optimizer):
        self.optimizer = optimizer
        self._step = 0
        self.warmup = warmup
        self.factor = factor
        self.model_size = model_size
        self._rate = 0
        
    def step(self):
        "Update parameters and rate"
        self._step += 1
        rate = self.rate()
        for p in self.optimizer.param_groups:
            p['lr'] = rate
        self._rate = rate
        self.optimizer.step()
        
    def rate(self, step = None):
        "Implement `lrate` above"
        if step is None:
            step = self._step
        return self.factor * \
            (self.model_size ** (-0.5) *
            min(step ** (-0.5), step * self.warmup ** (-1.5)))

def get_my_opt(model, learning_rate=2, warmup_steps=4000):
    # customized optimization
    return NoamOpt(model.src_embed[0].d_model, learning_rate, warmup_steps,
            torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))

class LabelSmoothing(nn.Module):
    "Implement label smoothing."
    def __init__(self, size, smoothing=0.0):
        super(LabelSmoothing, self).__init__()
        #  If the field size_average is set to False, the losses are instead summed for each minibatch.
        self.criterion = nn.KLDivLoss(size_average=False)
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.size = size
        self.true_dist = None
        
    def forward(self, x, target):
        assert x.size(2) == self.size
        true_dist = x.data.clone()
        # -1 for target positions
        true_dist.fill_(self.smoothing / (self.size - 1))
        # put confidence to target positions
        true_dist.scatter_(2, target.data.unsqueeze(2), self.confidence)

        self.true_dist = true_dist
        return self.criterion(x, Variable(true_dist, requires_grad=False))

def greedy_decode(model, src, src_mask, max_len, greedy):
    memory = model.encode(src, src_mask)
    ys = torch.zeros(src.shape[0], 1).type_as(src.data)
    for i in range(max_len):
        out = model.decode(memory, src_mask, 
                           Variable(ys), # target
                           Variable(subsequent_mask(ys.size(1)).type_as(src.data))) # target mask
        log_prob = model.generator(out[:, -1])

        if greedy:
            _, next_word = torch.max(log_prob, dim = 1, keepdim=True)
        else:
            next_word = torch.multinomial(torch.exp(log_prob), 1)

        next_word = next_word.data
        ys = torch.cat([ys, next_word.type_as(src.data)], dim=1)

    return ys

def beam_decode(model, memory, src_mask, args, tokens=None): # batch beam search
    if tokens is None:
        decoder_input = torch.zeros(1).long().to(memory.device)
        decoder_input = decoder_input.repeat(memory.shape[0]).unsqueeze(1)
    else: # for Monte Carlo search case where some of tokens are given
        decoder_input = tokens.long().to(memory.device)

    sequences = decoder_input.unsqueeze(1) # batch x beam x seq
    logps = torch.zeros((decoder_input.shape[0], args.beam_width)).type_as(memory) # batch x beam

    seq_len = sequences.shape[-1]
    # start beam search
    while args.max_target_len+1 > seq_len:
        decoder_inputs = sequences.view(-1,seq_len)
        # repeat: 0,1,2,0,1,2 repeat_interleave: 0,0,1,1,2,2
        out = model.decode(memory.repeat_interleave(sequences.shape[1], dim=0), src_mask,
                           decoder_inputs, subsequent_mask(seq_len).type_as(memory.data))
        log_prob = model.generator(out[:,-1,:]) # use only last position

        if not torch.all(logps==0): # skip for the first loop
            log_prob = logps.view(-1,1) + log_prob
        log_prob = log_prob.view(sequences.shape[0],-1)
        # sample next beams based on log_prob
        indexes = torch.multinomial(torch.exp(log_prob/(seq_len+1)), args.beam_width) # rows of input do not need to sum to one
        log_prob = torch.gather(log_prob, dim=1, index=indexes)

        selection = (indexes // args.vocab_size).type_as(indexes) # check what sequence number is
        selection = selection.unsqueeze(-1).repeat(1,1,sequences.shape[-1])
        indices = indexes % args.vocab_size

        selective_sequences = torch.gather(sequences, dim=1, index=selection)
        sequences = torch.cat((selective_sequences, indices.unsqueeze(-1)), dim=-1)
        logps = log_prob
        seq_len += 1

    best_sequence = sequences[torch.arange(len(logps)), torch.max(logps,dim=1)[1]]
    return best_sequence, sequences.view(-1,seq_len)

def fix_random(random_seed):
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed) # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)
    random.seed(random_seed)
