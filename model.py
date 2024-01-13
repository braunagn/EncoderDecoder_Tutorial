# architecture diagram:
# https://docs.google.com/presentation/d/1sRWV0hxIgL8ZNyrqV_jz7vX5l815RJI5l_KlXoVJHAQ/

import config
import torch
import torch.nn as nn
import torch.nn.functional as F


class EnP(nn.Module):
    def __init__(self, vocab_size, C, T, dropout):
        # vocab_size: total tokens in vocab (see config)
        # C: embedding size
        # T: max sequence len
        # dropout: float, pct of neurons to ignore during update
        super().__init__()

        self.temb = nn.Embedding(vocab_size, C)
        self.pemb = nn.Embedding(T, C)
        self.lnorm = nn.LayerNorm(C)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        # in (B,T/t) -> (B,T/t,C)
        _, t = x.shape
        device = x.device.type
        if device == "cuda":
            device = x.device.index
        x1 = self.temb(x)
        x2 = self.pemb(torch.arange(t, device=device))
        return self.lnorm(self.dropout(x1 + x2))  # (B,T,C)


class Head(nn.Module):
    """scaled dot production attention"""

    # in: (B,T/t,C) -> out: (B,T/t,hs)
    def __init__(self, head_size, C, T, self_mask=False):
        # head size: should be C // num_heads (see config)
        # C: embedding size
        # T: max sequence len
        # self_mask: whether or not to apply masked self-attention
        super().__init__()

        self.head_size = head_size
        self.C = C
        self.self_mask = self_mask
        self.register_buffer("tril", torch.tril(torch.ones(T, T)) == 0)

        self.query = nn.Linear(C, head_size, bias=False)
        self.key = nn.Linear(C, head_size, bias=False)
        self.value = nn.Linear(C, head_size, bias=False)

    def forward(self, V, K, Q, pad_mask=None):
        # Figure 2 (right side) from Attention paper
        # V, K, Q:  (B,T,C) Q of size (B,t,C) during inference
        # pad_mask: (B,1,T) from Dataset.__getitem__()

        q = self.query(Q)  # (B,T,HS); (B,t,HS) during inference within decoder only
        k = self.key(K)  # (B,T,HS); ^ same
        v = self.value(V)  # (B,T,HS); ^ same

        A = (
            q @ k.transpose(-2, -1) * self.C**-0.5
        )  # (B,T,hs) @ (B,hs,T) -> (B,T,T) training & inference within encoder only
        # (B,t,hs) @ (B,hs,T) -> (B,t,T) inference within decoder only

        _, t, T = A.shape

        if self.self_mask:
            # perform masked self attention
            A = A.masked_fill(self.tril[:t, :T], float("-inf"))  # (B,T,T); (B,t,T)
        if pad_mask is not None:
            # perform pad masking
            A = A.masked_fill(pad_mask, float("-inf")).masked_fill(
                pad_mask.transpose(-2, -1)[:, :t, :], -1e9
            )
        A = F.softmax(A, dim=-1)
        return A @ v  # (B,T,T) @ (B,T,hs) -> (B,T,hs) (encoder only)
        # (B,T/t,T) @ (B,T/t,hs) -> (B,T/t,hs) (decoder only)


class MultiHead(nn.Module):
    """concatenation of Heads with downstream linear projection"""

    def __init__(self, num_heads, head_size, C, T, self_mask, dropout):
        # num_heads: int, # of heads to include
        # # head size: should be C // num_heads (see config)
        # C: embedding size
        # T: max sequence len
        # self_mask: whether or not to apply masked self-attention
        # dropout: float, pct of neurons to ignore during update
        super().__init__()

        self.heads = nn.ModuleList(
            [Head(head_size, C, T, self_mask) for _ in range(num_heads)]
        )
        self.linear = nn.Linear(num_heads * head_size, C)
        self.lnorm = nn.LayerNorm(C)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, V, K, Q, pad_mask=None):
        x1 = torch.cat(
            [h(V, K, Q, pad_mask) for h in self.heads], dim=-1
        )  # (B,T,nh*hs); (B,1,nh*hs)
        x = Q + self.dropout(self.linear(x1))  # (B,T,C); (B,1,C)
        return self.lnorm(x)


class FeedForwardHead(nn.Module):
    """fully connected network + add + norm"""

    def __init__(self, C, dropout):
        # C: embedding size
        # dropout: float, pct of neurons to ignore during update
        super().__init__()

        self.FF = nn.Sequential(
            # adjust linear sizes as desired
            nn.Linear(C, 4 * C),
            nn.ReLU(),
            nn.Linear(4 * C, C),
        )
        self.lnorm = nn.LayerNorm(C)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        x = x + self.dropout(self.FF(x))
        return self.lnorm(x)


class EncoderLayer(nn.Module):
    """self attention + feedforward 'head'"""

    # in: (B,T,C) -> out: (B,T,C)
    def __init__(self, num_heads, head_size, C, T, dropout):
        # num_heads: int, # of heads to include
        # # head size: should be C // num_heads (see config)
        # C: embedding size
        # T: max sequence len
        # dropout: float, pct of neurons to ignore during update
        super().__init__()
        self.self_attn = MultiHead(
            num_heads, head_size, C, T, self_mask=False, dropout=dropout
        )  # always self_mask=False for encoder
        self.FFhead = FeedForwardHead(C, dropout)

    def forward(self, x, pad_mask):
        x = self.self_attn(x, x, x, pad_mask)  # V, K, Q = x
        x = self.FFhead(x)
        return x


class Encoder(nn.Module):
    """multipler layers of EncoderLayer"""

    # in: (B,T,C) -> out: (B,T,C)
    def __init__(self, n_layers, num_heads, head_size, C, T, dropout):
        # n_layers: number of EncoderLayers to include within Encoder
        # num_heads: int, # of heads to include
        # head size: should be C // num_heads (see config)
        # C: embedding size
        # T: max sequence len
        # dropout: float, pct of neurons to ignore during update
        super().__init__()
        self.layers = nn.ModuleList(
            [EncoderLayer(num_heads, head_size, C, T, dropout) for _ in range(n_layers)]
        )

    def forward(self, x, pad_mask):
        for layer in self.layers:
            x = layer(x, pad_mask)
        return x


class DecoderLayer(nn.Module):
    """masked self attention + cross attention + feed forward 'head'"""

    # in: (B,T/t,C) -> out: (B,T/t,C)
    def __init__(self, num_heads, head_size, C, T, dropout):
        # num_heads: int, # of heads to include
        # # head size: should be C // num_heads (see config)
        # C: embedding size
        # T: max sequence len
        # dropout: float, pct of neurons to ignore during update
        super().__init__()
        self.masked_self_attn = MultiHead(
            num_heads, head_size, C, T, self_mask=True, dropout=dropout
        )  # mask first sublayer of decoderlayer
        self.cross_attn = MultiHead(
            num_heads, head_size, C, T, self_mask=False, dropout=dropout
        )  # dont mask cross attention sublayer of decoderlayer
        self.FFhead = FeedForwardHead(C, dropout)

    def forward(self, Ve, Ke, Q, pad_mask=None):
        # Ve = Ke = output from Encoder; Q = output from prev step (DecoderLayer or EnP)
        Q = self.masked_self_attn(Q, Q, Q, pad_mask)
        Q = self.cross_attn(Ve, Ke, Q, pad_mask)
        return self.FFhead(Q)


class Decoder(nn.Module):
    """multipler layers of DecoderLayer"""

    # in: (B,T/t,C) -> out: (B,T/t,C)
    def __init__(self, n_layers, num_heads, head_size, C, T, dropout):
        # n_layers: number of EncoderLayers to include within Encoder
        # num_heads: int, # of heads to include
        # head size: should be C // num_heads (see config)
        # C: embedding size
        # T: max sequence len
        # dropout: float, pct of neurons to ignore during update
        super().__init__()
        self.layers = nn.ModuleList(
            [DecoderLayer(num_heads, head_size, C, T, dropout) for _ in range(n_layers)]
        )

    def forward(self, Ve, Ke, Q, pad_mask=None):
        for layer in self.layers:
            Q = layer(Ve, Ke, Q, pad_mask)
        return Q


class LanguageModel(nn.Module):
    """highest level class wrapper representing entire transformer model; params from config.py

    see here for visual of architecture layout:
    https://docs.google.com/presentation/d/1sRWV0hxIgL8ZNyrqV_jz7vX5l815RJI5l_KlXoVJHAQ/edit#slide=id.g29d8fe39d9a_0_0
    """

    def __init__(self, vocab_len):
        super().__init__()

        # architecture:
        self.encoder_embeddings = EnP(vocab_len, config.C, config.T, config.DROPOUT)
        self.decoder_embeddings = EnP(vocab_len, config.C, config.T, config.DROPOUT)
        self.encoder = Encoder(
            config.N_LAYERS,
            config.NUM_HEADS,
            config.HEAD_SIZE,
            config.C,
            config.T,
            config.DROPOUT,
        )
        self.decoder = Decoder(
            config.N_LAYERS,
            config.NUM_HEADS,
            config.HEAD_SIZE,
            config.C,
            config.T,
            config.DROPOUT,
        )
        self.linear = nn.Linear(config.C, vocab_len)

    def forward(self, x1, x2, x1padmask):
        # x1: sequence of NL to translate, x2: target (sequence of EN)
        # x1padmask: indicates where padding is within x1 (so it can be ignored)
        x1 = self.encoder_embeddings(x1)
        x2 = self.decoder_embeddings(x2)
        x1 = self.encoder(x1, x1padmask)  # V, K, Q = x1 (all same for encoder)
        x2 = self.decoder(x1, x1, x2)
        return self.linear(x2)  # no softmax (accounted for in loss:CrossEntropyLoss)



    # # helper functions for inference
    # def encoded_input_and_padmask(self, s):
    #     s = self.BOS_TOKEN + s + self.EOS_TOKEN
    #     x1 = torch.tensor([self.encode_sentence(s).ids], device=config.DEVICE)
    #     x1padmask = x1 == self.token_to_id(self.PAD_TOKEN)
    #     return x1, x1padmask[:, None, :]

    # def cleanup(self, s):
    #     remove = ["Ġ", "ġ", " ##"]
    #     for c in remove:
    #         s = s.replace(c, "")
    #     return s.replace(" .", ".")

    # # inference function (translates NL -> EN)
    # def generate(self, s, max_len=None, greedy=True):
    #     # s: NL sentence to translate (str)
    #     # max_len: override if you want len <T
    #     # greedy: use multinomal sampling of vocab or argmax.  recommend argmax for this project.
    #     max_len = config.T if max_len is None else max_len
    #     eos_token_id = self.token_to_id(self.EOS_TOKEN)
    #     x1, x1padmask = self.encoded_input_and_padmask(s)
    #     x2 = torch.tensor(
    #         [[self.token_to_id(self.BOS_TOKEN)]], device=config.DEVICE
    #     )

    #     for _ in range(max_len):
    #         # model prediction and probs
    #         logits = self.forward(x1, x2, x1padmask)[
    #             :, -1, :
    #         ]  # only last token (B,t,vl) -> (1,vl)
    #         probs = F.softmax(logits, dim=-1)
    #         if greedy:
    #             id = probs.argmax().expand(1, 1)
    #         else:
    #             id = torch.multinomial(probs, num_samples=1)
    #         x2 = torch.cat([x2, id], dim=-1)  # out_ids.append(out)
    #         if id == eos_token_id:
    #             break
    #     raw_sentence = self.decode_to_sentence(x2[0].detach().tolist())
    #     return self.cleanup(raw_sentence)
