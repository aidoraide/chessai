from utils.neural_utils import ResBlock, Flatten, piece2plane
from utils import train_utils, data_utils, constants
import chess
from chess import Board, Move, ROOK, BISHOP, KNIGHT, QUEEN, PAWN, KING, WHITE, BLACK
import pandas as pd
import numpy as np
import re
import itertools
import os
import time
import math

from collections import OrderedDict, Counter, defaultdict
from multiprocessing.pool import Pool

import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
device = torch.device("cuda:0")


class ChessNet(nn.Module):
    def __init__(self, n_res_blocks=19, learning_rate=0.01, bias=False, gpu_id=0):
        super(ChessNet, self).__init__()
        res_blocks = [(f'res_block{i+1}', ResBlock())
                      for i in range(n_res_blocks)]
        self.res_tower = nn.Sequential(OrderedDict([
            ('conv1', nn.Conv2d(len(piece2plane), 256, 3, padding=1)),
            ('batchnorm1', nn.BatchNorm2d(256)),
            ('relu1', nn.ReLU(inplace=True)),
            *res_blocks,
        ]))
        self.policy_head = nn.Sequential(OrderedDict([
            ('conv1', nn.Conv2d(256, 256, 3, padding=1)),
            ('batchnorm1', nn.BatchNorm2d(256)),
            ('relu1', nn.ReLU(inplace=True)),
            ('conv2', nn.Conv2d(256, 73, 1)),
            ('flatten', Flatten()),
            ('softmax', nn.Softmax(dim=1)),
        ]))
        self.value_head = nn.Sequential(OrderedDict([
            ('conv256+x_1', nn.Conv2d(256, 1, 1)),
            ('batchnorm256_1', nn.BatchNorm2d(1)),
            ('relu1', nn.ReLU(inplace=True)),
            ('flatten', Flatten()),
            ('fc64_256', nn.Linear(64, 256)),
            ('relu3', nn.ReLU(inplace=True)),
            # ('dropout', nn.Dropout(p=0.7)),
            ('fc256_1', nn.Linear(256, 1)),
            ('tanh', nn.Tanh()),
            # ('softmax', nn.Softmax(dim=1)),
        ]))

    def forward(self, x):
        tower_out = self.res_tower(x)
        policy_out = self.policy_head(tower_out)
        value_out = self.value_head(tower_out)
        # value_out = (value_out**2)/(value_out**2).sum(dim=1).view(-1, 1) # L2 Norm

        return policy_out.view(-1, 73, 8, 8), value_out.view(-1, 1)


def init_weights(m):
    classname = m.__class__.__name__
    if 'Linear' in classname:
        bound = 1.0 / m.weight.data.shape[1]
        m.weight.data.uniform_(-bound, bound)
        nn.init.zeros_(m.bias)
    # if 'Conv2d' in classname:
    #     m.weight.data.normal_(0, 1)
    # if 'BatchNorm2d' in classname:
    #     m.weight.data.normal_(0, 1)


if __name__ == '__main__':
    if not os.path.exists('models'):
        os.mkdir('models')

    BATCH_SIZE = 1024
    N_CONV_LAYERS = 24
    LOSS_REDUCTION = 'sum'
    policy_criterion = nn.BCELoss(reduction=LOSS_REDUCTION)
    value_criterion = nn.MSELoss(reduction=LOSS_REDUCTION)
    VALUE_WEIGHT = 1
    POLICY_WEIGHT = 1

    nnet = ChessNet(N_CONV_LAYERS)
    nnet.to(device)
    nnet.apply(init_weights)
    optimizer = optim.SGD(nnet.parameters(), lr=5e-5,
                          momentum=0.9, weight_decay=1e-5, nesterov=True)
    train_dl, val_dl, test_dl = data_utils.get_split_dataloaders(
        BATCH_SIZE, data_utils.get_lichess_dataframe)
    scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[int(i*len(train_dl)/2) for i in range(1, 4)])

    h_params = {
        'batch_size': BATCH_SIZE,
        'conv_layers': N_CONV_LAYERS,
        'reduction': LOSS_REDUCTION,
        'value_weight': VALUE_WEIGHT,
        'policy_weight': POLICY_WEIGHT,
        'optimizer': {**optimizer.state_dict(), 'params': None, 'class': optimizer.__class__.__module__},
        'scheduler': {**scheduler.state_dict(), 'class': scheduler.__class__.__module__},
    }

    writer = SummaryWriter()
    # writer.add_hparams(h_params)

    # if os.path.exists(constants.MODEL_PATH):
    #     nnet.load_state_dict(torch.load(constants.MODEL_PATH))

    EPOCHS = 4

    best_loss = float('inf')

    for epoch in range(EPOCHS):
        val_loss = train_utils.fit_epoch(nnet, optimizer, scheduler, policy_criterion,
                                         value_criterion, POLICY_WEIGHT, VALUE_WEIGHT, train_dl, val_dl, writer, epoch=epoch)
        if val_loss < best_loss:
            best_loss = val_loss
            print('Saving model!')
            torch.save(nnet.state_dict(), constants.MODEL_PATH)
        # scheduler.step()

    nnet.eval()
    p_correct, values, value_losses, value_exp, total, value_loss_total, policy_loss_total = 0, [
    ], [], [], 0, 0.0, 0.0
    with torch.set_grad_enabled(False):
        for i, batch in enumerate(test_dl):
            # Transfer to GPU
            state, policy, value = batch['state'].to(
                device), batch['policy'].to(device), batch['value'].to(device)
            policy_out, value_out = nnet(state)
            policy_loss = policy_criterion(policy_out, policy)
            value_loss = value_criterion(value_out, value)

            p_correct += sum(policy.flatten(1).argmax(1) ==
                             policy_out.flatten(1).argmax(1)).item()
            total += policy.size()[0]
            value_loss_total += value_loss.item()
            policy_loss_total += policy_loss.item()

            value_losses.append(value_loss.item())
            values.extend([v.item() for v in value_out])
            value_exp.extend([v.item() for v in value])
