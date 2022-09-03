import math
import torch
import torch.nn as nn

"""
Captcha reader from a tensor derived from an image, where
input image size is assumed to be w x h = 60 x 30, output is fixed at 5 characters
consists of a 1D-CNN and 5-headed attention

Args:
    num_labels (int): number of labels, default = 36
    window_size (int): window (convolutional kernel) size, default = 6
    hidden_size (int): hidden layer size (CNN output channel and attention input), default = 64
    dropout (float): dropout coefficient, default = 0.1
"""
class SimpleCaptchaReader(nn.Module):
    def __init__(self, num_labels: int = 36, window_size: int = 6, hidden_size: int = 64, dropout: float = 0.1):
        super(SimpleCaptchaReader, self).__init__()
        self.conv = nn.Conv1d(30, hidden_size, window_size)  # output size: batch_size x 55 x hidden_size
        self.pos_enc = PositionalEncoding(hidden_size, dropout)
        self.layernorm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(p=dropout)

        self.attn = nn.Linear(hidden_size, 5)  # output: batch_size x 55 x 5 -> softmaxed over dim=1
                                               # -> bmm output: batch_size x 5 x hidden_size
        self.out = nn.Linear(hidden_size, num_labels)

    def forward(self, inp: torch.Tensor, is_logits=False):
        """ Forward pass: conv -> pos_enc -> layernorm -> dropout -> attn -> dropout
        """
        conv_out = self.conv(inp.transpose(1, 2)).transpose(1,2).contiguous()
        conv_out = self.dropout(self.layernorm(conv_out))  # batch_size x 55 x hidden_size
        pos_enc = self.pos_enc(torch.zeros_like(conv_out))
        attn_probs = self.dropout(torch.softmax(self.attn(pos_enc), dim=1))  # batch_size x 55 x 5
        attn_out = self.dropout(torch.bmm(attn_probs.transpose(1, 2), conv_out))  # batch_size x 5 x hidden_size
        out = self.out(attn_out)  # batch_size x 5 x num_labels
        if is_logits:
            return out
        return torch.softmax(out, 2)

"""
Positional encoding based on sinusoidal values (Vaswani et al., NIPS 2017)

Args:
    dim_size (int): dimension size
    dropout (float): dropout coefficient, default = 0.1
"""
class PositionalEncoding(nn.Module):
    def __init__(self, dim_size, dropout=0.1):
        super(PositionalEncoding, self).__init__()

        pe = torch.zeros(55, dim_size)
        position = torch.arange(0, 55, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim_size, 2).float() * (-math.log(10000.0) / dim_size))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.pe = nn.Embedding.from_pretrained(pe, freeze=True)
    
    def forward(self, x: torch.Tensor):
        pids = torch.arange(x.size(1), dtype=torch.long, device=x.device)
        x = x + self.pe(pids)
        return x
