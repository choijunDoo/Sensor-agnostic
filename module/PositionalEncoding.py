import math

import torch
from torch import nn, Tensor

device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')

class PositionalEncoding(nn.Module):
    def __init__(self, dim_model, max_seq_len=130, dropout=0.1):
        super(PositionalEncoding, self).__init__()

        self.dropout = nn.Dropout(p=dropout)
        self.encoding = torch.zeros(max_seq_len, dim_model, device=device)
        self.encoding.requires_grad = False

        pos = torch.arange(0, max_seq_len, device=device)
        pos = pos.float().unsqueeze(dim=1)

        _2i = torch.arange(0, dim_model, step=2, device=device).float()

        self.encoding[:,0::2] = torch.sin(pos / (10000 ** (_2i / dim_model)))
        self.encoding[:,1::2] = torch.cos(pos / (10000 ** (_2i / dim_model)))

    def forward(self, x):
        batch_size, _, seq_len = x.size() ## (bs, 512, 130(sequence + class token))
        enc = self.encoding[:seq_len, :] ## (130, 512)
        x = x.permute(0,2,1)
        x = x + enc ## (bs,130,512) + (130,512)

        return self.dropout(x)