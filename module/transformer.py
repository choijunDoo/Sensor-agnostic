from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch import nn
import torch.nn.functional as F

from module.PositionalEncoding import PositionalEncoding

class Transformer(nn.Module):
    def __init__(self,
                 input_channel=14,
                 max_seq_len=513,
                 dim_model=512,
                 n_head=8,
                 hidden_dim=2048,
                 num_layers=6,
                 dropout=0.1):
        super(Transformer, self).__init__()

        self.dim_model = dim_model
        self.emb = nn.Conv1d(input_channel, dim_model, 3, stride=1, padding=1)
        self.pos_enc = PositionalEncoding(dim_model=dim_model,
                                          max_seq_len=max_seq_len,
                                          dropout=dropout)
        self.encoder_layers = TransformerEncoderLayer(d_model=dim_model,
                                                      nhead=n_head,
                                                      dim_feedforward=hidden_dim,
                                                      dropout=dropout)
        self.encoder = TransformerEncoder(self.encoder_layers,
                                          num_layers)
        self.decoder = nn.Linear(dim_model, 1)
        self.sigmoid = nn.Sigmoid()

        self.init_weights()

    def forward(self, src):
        src = self.emb(src)
        src = self.pos_enc(src)

        output = self.encoder(src)
        output = output[:,0,:]
        output = self.decoder(output)

        return self.sigmoid(output)

    def init_weights(self):
        for n, p in self.named_parameters():
            if p.data.ndimension() >= 2:
                nn.init.xavier_uniform_(p.data)