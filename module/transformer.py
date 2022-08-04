from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch import nn
import torch.nn.functional as F
import torch

from module.PositionalEncoding import PositionalEncoding

device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')

class Transformer(nn.Module):
    def __init__(self,
                 input_channel=14,
                 max_seq_len=513,
                 dim_model=512,
                 n_head=8,
                 hidden_dim=1024,
                 num_layers=6,
                 dropout=0.1):
        super(Transformer, self).__init__()

        self.dim_model = dim_model
        self.emb = nn.Conv1d(input_channel, dim_model, 3, 1, 1)

        self.pos_enc = PositionalEncoding(dim_model=dim_model,
                                          max_seq_len=max_seq_len,
                                          dropout=dropout)

        self.encoder_layers = TransformerEncoderLayer(d_model=dim_model,
                                                      nhead=n_head,
                                                      dim_feedforward=hidden_dim,
                                                      dropout=dropout)

        self.encoder = TransformerEncoder(self.encoder_layers, num_layers)

        self.fc1 = nn.Linear(dim_model, 256)
        self.fc2 = nn.Linear(256, 16)
        self.relu = nn.ReLU()

        self.decoder = nn.Linear(16, 1)

        self.init_weights()

    def forward(self, src):
        src = src.permute(0, 2, 1)  ## (bs, c, s)
        src = self.emb(src)

        cls_token = torch.zeros(src.shape[0], src.shape[1], 1).to(device)
        src = torch.cat((cls_token, src), 2) # (bs, c, s+1)

        src = self.pos_enc(src) # (bs, s, c)

        output = self.encoder(src)
        output = output[:,0,:]
        output = self.relu(self.fc1(output))
        output = self.relu(self.fc2(output))
        output = self.decoder(output)

        return output

    def init_weights(self):
        for n, p in self.named_parameters():
            if p.data.ndimension() >= 2:
                nn.init.xavier_uniform_(p.data)