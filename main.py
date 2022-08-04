import os
import random

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import time, datetime
import argparse
import numpy as np
from pathlib import Path

import utils, dataloader
from torch.optim import SGD, Adam

from module.transformer import Transformer

device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')

def train(tr_dataloader, ts_dataloader, epochs, model, criterion, args):
    optimizer = Adam(params=model.parameters(),
                     lr=args.lr,
                     weight_decay=args.weight_decay)

    model.train()
    tr_accuracy, ts_accuracy = [], []

    for epoch in range(epochs):
        tr_loss = 0
        cnt = 0
        correct = 0

        for idx, (data, target) in enumerate(tr_dataloader):
            # data = data.permute(0, 2, 1)
            data = data[:,:,:14]

            data, target = data.to(device), target.to(device)

            target[target < 5] = 0
            target[target >= 5] = 1

            outputs = model(data).squeeze()
            loss = criterion(outputs, target)
            tr_loss += loss.item()

            loss.backward()
            # torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_norm)
            optimizer.step()

            model.zero_grad()

            outputs[outputs >= 0.5] = 1
            outputs[outputs < 0.5] = 0

            correct += (target == outputs).float().sum()
            cnt += target.shape[0]

            print("\r[epoch {:3d}/{:3d}] [batch {:4d}/{:4d}] loss: {:.6f} acc: {:.4f}".format(
                epoch, args.n_epochs, idx + 1, len(tr_dataloader), tr_loss / (idx+1), correct / cnt), end=' ')

        tr_accuracy.append(correct / cnt)

        model.eval()

        ts_loss = 0
        ts_correct = 0
        ts_cnt = 0

        with torch.no_grad():
            for idx, (data, target) in enumerate(ts_dataloader):
                # data = data.permute(0, 2, 1)
                data = data[:,:,:14]

                data, target = data.to(device), target.to(device)

                target[target < 5] = 0
                target[target >= 5] = 1

                outputs = model(data).squeeze()
                loss = criterion(outputs, target)

                ts_loss += loss.item()

                outputs[outputs >= 0.5] = 1
                outputs[outputs < 0.5] = 0

                ts_correct += (target == outputs).float().sum()
                ts_cnt += target.shape[0]

                print("\r[epoch {:3d}/{:3d}] [batch {:4d}/{:4d}] loss: {:.6f} acc: {:.4f}".format(
                    epoch, args.n_epochs, idx + 1, len(ts_dataloader), ts_loss / (idx+1), ts_correct / ts_cnt), end=' ')

        print(ts_correct / ts_cnt)
        ts_accuracy.append(ts_correct / ts_cnt)

    tr_loss /= cnt
    tr_acc = correct / cnt

    return tr_loss, tr_acc, tr_accuracy, ts_accuracy

def main():
    parser = argparse.ArgumentParser(description='NMT - Transformer')
    """ recommend to use default settings """

    # environmental settings
    parser.add_argument('--gpu_id', type=int, default=0)
    parser.add_argument('--save', action='store_true', default=0)

    # architecture
    # parser.add_argument('--modal', type=int, default="all")
    parser.add_argument('--model_dim', type=int, default=512, help='Dimension size of model dimension')
    parser.add_argument('--hidden_size', type=int, default=2048, help='Dimension size of hidden states')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout probability')
    parser.add_argument('--n_head', type=int, default=8, help='Number of multi-head Attention')
    parser.add_argument('--num_layers', type=int, default=6, help='Number of Encoder layers')
    parser.add_argument('--max_norm', type=float, default=5.0)


    # hyper-parameters
    parser.add_argument('--n_epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--warmup_steps', type=int, default=78, help='Warmup step for scheduler')

    args = parser.parse_args()

    tr_dataset = dataloader.AMIGOSDataset(is_train=True)
    ts_dataset = dataloader.AMIGOSDataset(is_train=False)

    tr_dataloader = DataLoader(tr_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True, num_workers=0)
    ts_dataloader = DataLoader(ts_dataset, batch_size=args.batch_size, shuffle=False, drop_last=False,num_workers=0)

    ## 여기 수정해야함 (모달리티 별로 Input Channel)
    model = Transformer(input_channel=14,
                        dim_model=args.model_dim,
                        n_head=args.n_head,
                        hidden_dim=args.hidden_size,
                        num_layers=args.num_layers,
                        dropout=args.dropout)

    model = model.to(device)
    criterion = nn.BCEWithLogitsLoss()

    tr_loss, tr_acc, tr_accuracy, ts_accuracy = train(tr_dataloader, ts_dataloader, args.n_epochs, model, criterion, args)
    print("tr: ({:.4f}, {:5.2f} | ".format(tr_loss, tr_acc * 100), end='')

    print(tr_accuracy, ts_accuracy)

    torch.save(model.state_dict(), "./model/model.pt")

    # pred = eval(ts_dataloader, model=model, args=args)
    # pred_filepath = './data/pred_real_test.npy'
    # np.save(pred_filepath, np.array(pred))

if __name__ == "__main__":
    main()