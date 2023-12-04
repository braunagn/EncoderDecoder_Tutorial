# architecture diagram:
# https://docs.google.com/presentation/d/1sRWV0hxIgL8ZNyrqV_jz7vX5l815RJI5l_KlXoVJHAQ/

import torch
import torch.nn as nn
import torch.nn.functional as F


class EnP(nn.Module):
    def __init__(self, vocab_size, C, T, dropout):
        super().__init__()

        self.temb = nn.Embedding(vocab_size, C)
        self.pemb = nn.Embedding(T, C)
        self.lnorm = nn.LayerNorm(C)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        # in (B,T/t) -> (B,T/t,C)
        _, t = x.shape
        x1 = self.temb(x)
        x2 = self.pemb(torch.arange(t, device=config.DEVICE))
        return self.lnorm(self.dropout(x1 + x2))  # (B,T,C)


class Head(nn.Module):
    """ scaled dot production attention """
    # in: (B,T/t,C) -> out: (B,T/t,hs)
    def __init__(self, head_size, C, T, self_mask=False):
        super().__init__()

        self.head_size = head_size
        self.C = C
        self.self_mask = self_mask
        self.register_buffer("tril", torch.tril(torch.ones(T, T)) == 0)

        self.query = nn.Linear(C, head_size, bias=False)
        self.key = nn.Linear(C, head_size, bias=False)
        self.value = nn.Linear(C, head_size, bias=False)

    def forward(self, V, K, Q, pad_mask=None):
        # V, K, Q:  (B,T,C) Q of size (B,t,C) during inference
        # pad_mask: (B,1,T) from Dataset.__getitem__()

        q = self.query(Q)  # (B,T,HS); (B,t,HS) during inference within decoder only
        k = self.key(K)    # (B,T,HS); ^ same
        v = self.value(V)  # (B,T,HS); ^ same

        A = q @ k.transpose(-2, -1) * self.C**-0.5  # (B,T,hs) @ (B,hs,T) -> (B,T,T) training & inference within encoder only
                                                    # (B,t,hs) @ (B,hs,T) -> (B,t,T) inference within decoder only
        _, t, T = A.shape

        if self.self_mask:
            A = A.masked_fill(self.tril[:t, :T], float("-inf")) # (B,T,T); (B,t,T)
        if pad_mask is not None:
            A = A.masked_fill(pad_mask, float("-inf")).masked_fill(pad_mask.transpose(-2, -1)[:,:t,:], -1e9)
        A = F.softmax(A, dim=-1)
        return A @ v  # (B,T,T) @ (B,T,hs) -> (B,T,hs) training & inference within encoder only
                      # (B,t,T) @ (B,T,hs) -> (B,t,hs) inference within decoder only


class MultiHead(nn.Module):
    """ concatenation of Heads with downstream linear projection """
    def __init__(self, num_heads, head_size, C, T, self_mask, dropout):
        # `heads` is nn.ModuleList of Heads()
        super().__init__()

        self.heads = nn.ModuleList([Head(head_size, C, T, self_mask) for _ in range(num_heads)])
        self.linear = nn.Linear(num_heads * head_size, C)
        self.lnorm = nn.LayerNorm(C)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, V, K, Q, pad_mask=None):
        x1 = torch.cat([h(V, K, Q, pad_mask) for h in self.heads], dim=-1)  # (B,T,nh*hs); (B,1,nh*hs)
        x = Q + self.dropout(self.linear(x1))  # (B,T,C); (B,1,C)
        return self.lnorm(x)


class FeedForwardHead(nn.Module):
    """ fully connected network + add + norm """
    def __init__(self, C, dropout):
        super().__init__()

        self.FF = nn.Sequential(
            # adjust linear sizes as desired
            nn.Linear(C, 4*C),
            nn.ReLU(),
            nn.Linear(4*C, C)
        )
        self.lnorm = nn.LayerNorm(C)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        x = x + self.dropout(self.FF(x))
        return self.lnorm(x)


class EncoderLayer(nn.Module):
    """ self attention + feedforward 'head' """
    # in: (B,T,C) -> out: (B,T,C)
    def __init__(self, num_heads, head_size, C, T, dropout):
        super().__init__()
        self.self_attn = MultiHead(num_heads, head_size, C, T, self_mask=False, dropout=dropout)
        self.FFhead = FeedForwardHead(C, dropout)

    def forward(self, x, pad_mask):
        x = self.self_attn(x, x, x, pad_mask)  # V, K, Q = x
        x = self.FFhead(x)
        return x


class Encoder(nn.Module):
    """ multipler layers of EncoderLayer """
    # in: (B,T,C) -> out: (B,T,C)
    def __init__(self, n_layers, num_heads, head_size, C, T, dropout):
        super().__init__()
        self.layers = nn.ModuleList([EncoderLayer(num_heads, head_size, C, T, dropout) for _ in range(n_layers)])

    def forward(self, x, pad_mask):
        for layer in self.layers:
            x = layer(x, pad_mask)
        return x


class DecoderLayer(nn.Module):
    """ masked self attention + cross attention + feed forward 'head' """
    # in: (B,T/t,C) -> out: (B,T/t,C)
    def __init__(self, num_heads, head_size, C, T, dropout):
        super().__init__()
        self.masked_self_attn = MultiHead(num_heads, head_size, C, T, self_mask=True, dropout=dropout)
        self.cross_attn = MultiHead(num_heads, head_size, C, T, self_mask=False, dropout=dropout)
        self.FFhead = FeedForwardHead(C, dropout)

    def forward(self, Ve, Ke, Q, pad_mask=None):
        # Ve = Ke = output from Encoder; Q = output from prev step (DecoderLayer or EnP)
        Q = self.masked_self_attn(Q, Q, Q, pad_mask)
        Q = self.cross_attn(Ve, Ke, Q, pad_mask)
        return self.FFhead(Q)


class Decoder(nn.Module):
    """ multipler layers of DecoderLayer """
    # in: (B,T/t,C) -> out: (B,T/t,C)
    def __init__(self, n_layers, num_heads, head_size, C, T, dropout):
        super().__init__()
        self.layers = nn.ModuleList([DecoderLayer(num_heads, head_size, C, T, dropout) for _ in range(n_layers)])

    def forward(self, Ve, Ke, Q, pad_mask=None):
        for layer in self.layers:
            Q = layer(Ve, Ke, Q, pad_mask)
        return Q


class LanguageModel(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.config = config
        self.token_to_id = config.tokenizer.token_to_id
        self.id_to_token = config.tokenizer.id_to_token
        self.encode_sentence = config.tokenizer.encode
        self.decode_to_sentence = config.tokenizer.decode
        # architecture:
        self.encoder_embeddings = EnP(config.vocab_len, config.C, config.T, config.dropout)
        self.decoder_embeddings = EnP(config.vocab_len, config.C, config.T, config.dropout)
        self.encoder = Encoder(
            config.n_layers,
            config.num_heads,
            config.head_size,
            config.C,
            config.T,
            config.dropout,
        )
        self.decoder = Decoder(
            config.n_layers,
            config.num_heads,
            config.head_size,
            config.C,
            config.T,
            config.dropout,
        )
        self.linear = nn.Linear(config.C, config.vocab_len)

    def forward(self, x1, x2, x1padmask):
        # x1: sequence of NL to translate, x2: target, sequence of EN
        x1 = self.encoder_embeddings(x1)
        x2 = self.decoder_embeddings(x2)
        x1 = self.encoder(x1, x1padmask)  # V, K, Q = x1 (all same for encoder)
        x2 = self.decoder(x1, x1, x2)
        return self.linear(x2)  # return F.softmax(x, dim=-1)

    def encoded_input_and_padmask(self, s):
        s = self.config.BOS_TOKEN + s + self.config.EOS_TOKEN
        x1 = torch.tensor([self.encode_sentence(s).ids], device=self.config.DEVICE)
        x1padmask = x1==self.token_to_id(self.config.PAD_TOKEN)
        return x1, x1padmask[:,None,:]

    def cleanup(self, s):
        remove = ["Ġ", "ġ", " ##"]
        for c in remove:
            s = s.replace(c, "")
        return s

    def generate(self, s, max_len=None, greedy=False):
        # s: NL sentence to translate (str)
        max_len = self.config.T if max_len is None else max_len
        eos_token_id = self.token_to_id(self.config.EOS_TOKEN)
        x1, x1padmask = self.encoded_input_and_padmask(s)
        x2 = torch.tensor([[self.token_to_id(self.config.BOS_TOKEN)]], device=self.config.DEVICE)

        for _ in range(max_len):
            # model prediction and probs
            logits = self.forward(x1, x2, x1padmask)[:,-1,:]  # only last token (B,t,vl) -> (1,vl)
            probs = F.softmax(logits, dim=-1)
            if not greedy:
                id = torch.multinomial(probs, num_samples=1)
            else:
                id = probs.argmax().expand(1,1)
            x2 = torch.cat([x2, id], dim=-1)  # out_ids.append(out)
            if id == eos_token_id:
                break
        raw_sentence = self.decode_to_sentence(x2[0].detach().tolist())
        return self.cleanup(raw_sentence)



class Config(object):
    # model architecture and setup
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    vocab_len = len(vocab)
    n_layers = 6
    C = 512  # embedding dimension
    T = T  # context window (max tokens)
    num_heads = 8
    head_size = 512 // 8  # 64
    dropout = 0.1
    tokenizer = tokenizer
    BOS_TOKEN = "<s>"
    EOS_TOKEN = "</s>"
    PAD_TOKEN = "[PAD]"
    UNK_TOKEN = "[UNK]"


config = Config()

# def init_weights(m):
#     if not isinstance(m, (nn.Embedding, nn.LayerNorm, nn.Dropout)):
#         if m.dim() > 1:
#             torch.nn.init.xavier_uniform(m.weight)

# model = LanguageModel(config).to(config.DEVICE)
# model.apply(init_weights)
# for p in model.parameters():
#     if p.dim() > 1:
#         nn.init.xavier_uniform_(p)

model = torch.load(save_path_model_object, map_location=DEVICE)#.to(DEVICE)