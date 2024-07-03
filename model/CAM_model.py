
from torch import nn
import torch
import math
from torch.nn import functional as F
from torchtune.modules import RMSNorm, RotaryPositionalEmbeddings


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        scaled_hidden = int(2 / 3 * 4 * config.emb_dim)
        self.fc1 = nn.Linear(config.emb_dim, scaled_hidden, bias=False)
        self.fc2 = nn.Linear(config.emb_dim, scaled_hidden, bias=False)
        self.fc3 = nn.Linear(scaled_hidden, config.emb_dim, bias=False)

    def forward(self, x):
        x1 = self.fc1(x)
        x2 = self.fc2(x)
        hidden = F.silu(x1)
        hidden = hidden * x2
        return self.fc3(hidden)


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.emb_dim % config.n_head == 0
        self.emb_dim = config.emb_dim
        self.n_head = config.n_head
        self.head_dim = config.emb_dim // config.n_head
        self.batch_size = config.batch_size
        self.block_size = config.block_size

        self.Wq = nn.Linear(config.emb_dim, self.n_head * self.head_dim, bias=False)
        self.Wk = nn.Linear(config.emb_dim, self.n_head * self.head_dim, bias=False)
        self.Wv = nn.Linear(config.emb_dim, self.n_head * self.head_dim, bias=False)
        self.Wo = nn.Linear(config.emb_dim, self.n_head * self.head_dim, bias=False)
        self.pos_emb = RotaryPositionalEmbeddings(self.head_dim, config.block_size)

        self.cache_k = None
        self.cache_v = None


        # flash attention make GPU go brrrrr but support is only in PyTorch >= 2.0
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
        if not self.flash:
            print("WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0")
            # causal mask to ensure that attention is only applied to the left in the input sequence
            self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                                        .view(1, 1, config.block_size, config.block_size))
        

    def forward(self, x, start_pos):
        batch_size, seq_len, dim = x.shape
        assert dim == self.emb_dim, "dim must be equal to self.emb_dim"
        if start_pos == 0 or self.cache_k is None or self.cache_v is None:
            self.cache_k = torch.zeros((batch_size, self.block_size, self.n_head, self.head_dim), device=x.device)
            self.cache_v = torch.zeros((batch_size, self.block_size, self.n_head, self.head_dim), device=x.device)

        xq = self.Wq(x)
        xk = self.Wk(x)
        xv = self.Wv(x)

        xq = xq.view(batch_size, seq_len, self.n_head, self.head_dim)
        xk = xk.view(batch_size, seq_len, self.n_head, self.head_dim)
        xv = xv.view(batch_size, seq_len, self.n_head, self.head_dim)

        xq = self.pos_emb(xq)
        xk = self.pos_emb(xk)

        self.cache_k[:batch_size, start_pos:start_pos + seq_len] = xk
        self.cache_v[:batch_size, start_pos:start_pos + seq_len] = xv

        
        keys = self.cache_k[:batch_size, :start_pos + seq_len]
        values = self.cache_v[:batch_size, :start_pos + seq_len]
        
        queries = xq.transpose(1, 2)
        keys = keys.transpose(1, 2)
        values = values.transpose(1, 2)

        
         # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        if self.flash:
            # efficient attention using Flash Attention CUDA kernels
            context = torch.nn.functional.scaled_dot_product_attention(queries, keys, values, attn_mask=None, dropout_p=0 if self.training else 0, is_causal=True)
        else:
            # manual implementation of attention
            att = (queries @ keys.transpose(-2, -1)) * (1.0 / math.sqrt(keys.size(-1)))
            att = att.masked_fill(self.bias[:,:,:seq_len,:seq_len] == 0, float('-inf'))
            att = F.softmax(att, dim=-1)
            context = att @ values # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)


        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        output = self.Wo(context)
        return output


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.rn1 = RMSNorm(config.emb_dim)
        self.rn2 = RMSNorm(config.emb_dim)
        self.attn = MultiHeadSelfAttention(config)
        self.mlp = MLP(config)

    def forward(self, x, start_pos):
        x = x + self.attn(self.rn1(x), start_pos)
        x = x + self.mlp(self.rn2(x))
        return x


class CAM(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.inp_emb = nn.Embedding(config.vocab_size, config.emb_dim)
        self.blocks = nn.ModuleList([Block(config) for _ in range(config.n_layers)])
        self.fc_out = nn.Linear(config.emb_dim, config.vocab_size, bias=False)
        self.rmsnorm = RMSNorm(config.emb_dim)
        # self.inp_emb.weight = self.fc_out.weight # https://paperswithcode.com/method/weight-tying

    def forward(self, x, start_pos, y=None):
        batch, seq_len = x.shape
        x = self.inp_emb(x)
        for block in self.blocks:
            x = block(x, start_pos)
        x = self.rmsnorm(x)

        logits = self.fc_out(x)
        loss = None
        if y is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), y.view(-1), ignore_index=-1
            )

        return logits, loss

    @torch.no_grad()
    def generate(self, inp, temperature=1.0, top_k=10):
        inp = inp.reshape(1, -1)
        start_pos = 0
        for _ in range(self.config.block_size - inp.shape[1]):
            # print(start_pos)
            logits, _ = self.forward(inp[:, start_pos:], start_pos)
            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("Inf")
            probs = F.softmax(logits, dim=-1)
            inp_next = torch.multinomial(probs, num_samples=1)
            inp = torch.cat((inp, inp_next), dim=1)
            start_pos = inp.shape[0]

        return inp[0]

    @torch.no_grad()
    def generate_yield(self, inp, temperature=1.0, top_k=10):
        inp = inp.reshape(1, -1)
        start_pos = 0
        for _ in range(self.config.block_size - inp.shape[1]):
            # print(start_pos)
            logits, _ = self.forward(inp[:, start_pos:], start_pos)
            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("Inf")
            probs = F.softmax(logits, dim=-1)
            inp_next = torch.multinomial(probs, num_samples=1)
            inp = torch.cat((inp, inp_next), dim=1)
            start_pos = inp.shape[0]

            
            yield inp_next.item()
    