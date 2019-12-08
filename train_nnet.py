import chess
from chess import Board, Move, ROOK, BISHOP, KNIGHT, QUEEN, PAWN, KING, WHITE, BLACK
import pandas as pd
import numpy as np
import re, itertools, os, time, math

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

from utils import train_utils, data_utils, constants
from utils.neural_utils import ResBlock, Flatten, piece2plane

class ChessNet(nn.Module):
    def __init__(self, n_res_blocks=19, learning_rate=0.01, bias=False, gpu_id=0):
        super(ChessNet, self).__init__()
        res_blocks = [(f'res_block{i+1}', ResBlock()) for i in range(n_res_blocks)]
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

    writer = SummaryWriter()

    nnet = ChessNet(12)
    nnet.to(device)
    nnet.apply(init_weights)
    # if os.path.exists(constants.MODEL_PATH):
    #     nnet.load_state_dict(torch.load(constants.MODEL_PATH))

    BATCH_SIZE = 1024
    EPOCHS = 4
    policy_criterion = nn.BCELoss(reduction='sum')
    value_criterion = nn.MSELoss(reduction='sum')
    VALUE_WEIGHT = 1
    POLICY_WEIGHT = .5
    optimizer = optim.SGD(nnet.parameters(), lr=2e-5, momentum=0.9, weight_decay=1e-5, nesterov=True)
    train_dl, val_dl, test_dl = data_utils.get_split_dataloaders(BATCH_SIZE, data_utils.get_lichess_dataframe)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[int(i*len(train_dl)/3) for i in range(1, 4)])


    best_loss = float('inf')

    for epoch in range(EPOCHS):
        val_loss = train_utils.fit_epoch(nnet, optimizer, scheduler, policy_criterion, value_criterion, POLICY_WEIGHT, VALUE_WEIGHT, train_dl, val_dl, writer, epoch=epoch)
        if val_loss < best_loss:
            best_loss = val_loss
            print('Saving model!')
            torch.save(nnet.state_dict(), constants.MODEL_PATH)
        # scheduler.step()

    nnet.eval()
    p_correct, values, value_losses, value_exp, total, value_loss_total, policy_loss_total = 0, [], [], [], 0, 0.0, 0.0
    with torch.set_grad_enabled(False):
        for i, batch in enumerate(test_dl):
            # Transfer to GPU
            state, policy, value = batch['state'].to(device), batch['policy'].to(device), batch['value'].to(device)
            policy_out, value_out = nnet(state)
            policy_loss = policy_criterion(policy_out, policy)
            value_loss  = value_criterion(value_out, value)

            p_correct += sum(policy.flatten(1).argmax(1) == policy_out.flatten(1).argmax(1)).item()
            total += policy.size()[0]
            value_loss_total += value_loss.item()
            policy_loss_total += policy_loss.item()

            value_losses.append(value_loss.item())
            values.extend([v.item() for v in value_out])
            value_exp.extend([v.item() for v in value])

    # val_p_correct = sum([(vp >= 0 and ve >= 0) or (vp <= 0 and ve <= 0) for vp, ve in zip(values, value_exp) if ve != 0])*100./len(values)
    # val_p_close = sum([(vp >= -.05 and ve >= 0) or (vp <= .05 and ve <= 0) for vp, ve in zip(values, value_exp)])*100./len(values)
    # val_p_far = sum([(vp <= -.10 and ve >= 0) or (vp >= .10 and ve <= 0) for vp, ve in zip(values, value_exp)])*100./len(values)
    # val_p_correct_cls = sum([round(vp) == round(ve) for vp, ve in zip(values, value_exp)])*100./len(values)
    # print(f'{val_p_correct:.5f}% of values are predicted on the same side of 0 as expected.')
    # print(f'{val_p_close:.5f}% of values are predicted on the same side of 0 or CLOSE as expected.')
    # print(f'{val_p_correct_cls:.5f}% of values are predicted the same class as expected.')
    # print(f'{val_p_far:.5f}% of values are predicted FAR from the true label.')

    # matplotlib.rcParams['figure.figsize'] = [15, 10]

    # for i in range(4):
    #     filt = [-1, 0, 1] if i == 0 else [i-2]
    #     ax = plt.subplot(2, 2, i+1)
    #     ax.hist([v for v, exp in zip(values, value_exp) if exp in filt], bins=200, color='blue', edgecolor='black')
    #     ax.set_title(f'Histogram of predicted values in {filt}', size=20)
    #     ax.set_xlabel('Predicted value (win=1, loss=-1, tie=0)', size=17)
    #     ax.set_ylabel('Frequency', size=17)

    # plt.tight_layout()
    # plt.show()

    # matplotlib.rcParams['figure.figsize'] = [15, 10]

    # for i in range(4):
    #     filt = [-1, 0, 1] if i == 0 else [i-2]
    #     ax = plt.subplot(2, 2, i+1)
    #     ax.hist([v for v, exp in zip(values, value_exp) if exp in filt], bins=200, color='blue', edgecolor='black')
    #     ax.set_title(f'Histogram of predicted values in {filt}', size=20)
    #     ax.set_xlabel('Predicted value (win=1, loss=-1, tie=0)', size=17)
    #     ax.set_ylabel('Frequency', size=17)

    # plt.tight_layout()
    # plt.show()

    # ax = plt.subplot(1, 1, 1)
    # ax.hist(value_losses, bins=500, color='blue', edgecolor='black')
    # ax.set_title('Histogram of value losses', size=20)
    # ax.set_xlabel('Loss', size=17)
    # ax.set_ylabel('Frequency', size=17)

    # plt.tight_layout()
    # plt.show()

    if os.path.exists(constants.MODEL_PATH):
        model_state_dict = nnet.state_dict()
        saved_state_dict = torch.load(constants.MODEL_PATH)
        for name, param in saved_state_dict.items():
            if name not in model_state_dict:
                continue
            model_state_dict[name].copy_(param)

    for m in nnet.value_head:
        init_weights(m)
    # Train only the value head
    for param in nnet.parameters():
        param.requires_grad = False
    for param in nnet.value_head.parameters():
        param.requires_grad = True

    EPOCHS = 3
    BATCH_SIZE = 8192
    policy_criterion = nn.BCELoss(reduction='sum')
    value_criterion = nn.SmoothL1Loss(reduction='none')
    VALUE_WEIGHT = 1
    POLICY_WEIGHT = 1

    optimizer = optim.SGD(
        [
            {"params": nnet.res_tower.parameters(),   "lr": 0},
            {"params": nnet.policy_head.parameters(), "lr": 0},
            {"params": nnet.value_head.parameters(),  "lr": 1e-4},
        ],
        lr=0, momentum=0.9, weight_decay=1e-4
    )
    best_loss = float('inf')
    df = data_utils.get_dataframe()
    df = df.loc[df.value != 0] # remove ties
    train_dl, val_dl, test_dl = data_utils.get_dataloaders(BATCH_SIZE, df)

    for epoch in range(EPOCHS):
        val_loss = train_utils.fit_epoch(nnet, optimizer, scheduler, policy_criterion, value_criterion, POLICY_WEIGHT, VALUE_WEIGHT, train_dl, val_dl, epoch=epoch)
        if val_loss < best_loss:
            best_loss = val_loss
            print('Saving model!')
            torch.save(nnet.state_dict(), constants.MODEL_PATH)
        # scheduler.step()


    # nnet.eval()
    # p_correct, total, value_loss_total, policy_loss_total = 0, 0, 0., 0.
    # values, value_losses, value_exp, policy_losses, correct_policy_investment = [], [], [], [], []
    # with torch.set_grad_enabled(False):
    #     for i, batch in enumerate(test_dl):
    #         # Transfer to GPU
    #         state, policy, value = batch['state'].to(device), batch['policy'].to(device), batch['value'].to(device)
    #         policy_out, value_out = nnet(state)
    #         policy_loss = policy_criterion(policy_out, policy) * POLICY_WEIGHT
    #         value_loss  = value_criterion(value_out, value) * VALUE_WEIGHT

    #         p_correct += sum(policy.flatten(1).argmax(1) == policy_out.flatten(1).argmax(1)).item()
    #         total += policy.size()[0]
    #         value_loss_total += value_loss.item()
    #         policy_loss_total += policy_loss.item()

    #         value_losses.append(value_loss.item())
    #         policy_losses.append(policy_loss.item())
            
    #         values.extend([v.item() for v in value_out])
    #         value_exp.extend([v.item() for v in value])
    #         correct_policy_investment.extend([p.item() for p in (policy_out.flatten(1) * policy.flatten(1)).max(dim=1)[0]])
            
    # val_p_correct = sum([(vp >= 0 and ve >= 0) or (vp <= 0 and ve <= 0) for vp, ve in zip(values, value_exp)])*100./len(values)
    # val_p_close = sum([(vp >= -.05 and ve >= 0) or (vp <= .05 and ve <= 0) for vp, ve in zip(values, value_exp)])*100./len(values)
    # val_p_correct_cls = sum([round(vp) == round(ve) for vp, ve in zip(values, value_exp)])*100./len(values)
    # print(f'{val_p_correct:.5f}% of values are predicted on the same side of 0 as expected.')
    # print(f'{val_p_close:.5f}% of values are predicted on the same side of 0 or CLOSE as expected.')
    # print(f'{val_p_correct_cls:.5f}% of values are predicted the same class as expected.')

    # from chess import svg
    # val2color = lambda v, maxv: '#%02x%02x%02x%02x' % (int(v/maxv*255), 255 - int(v/maxv*255), 255 - int(v/maxv*255), int(v/maxv*255))
    # val2color = lambda v, maxv: '#%02x%02x%02x%02x' % (int(v/maxv*255), 0, 0, int(v/maxv*255))

# def visualize_nn_out(data_idx):
#     data = test_ds[data_idx]
#     state, value, policy = data['state'].to(device), data['value'].to(device), data['policy'].to(device)
#     raw = test_ds.get_raw(data_idx)
#     board, player_move = raw['board'], raw['move']
#     policy_out, value_out = nnet(state.unsqueeze(0))
#     print({WHITE: 'White', BLACK: 'Black'}[board.turn] + "'s turn")
#     print(f'Board evaluated to {[f"{v:.5f}" for v in value_out[0]]}, training value is {[f"{v:.5f}" for v in value]}')
#     moves_vals = sorted(((policy_out[0][z][x][y].item(), x, y, z, m) 
#                          for (x, y, z), m in ((move2idx(m), m) for m in board.legal_moves)), reverse=True)
#     arrows = []
#     maxv = moves_vals[0][0]
#     print(f'The most confident move (reddest) had a value of {maxv:.5f}')
#     for v, x, y, z, m in moves_vals:
#         arrows.append(svg.Arrow(m.from_square, m.to_square, color=val2color(v, maxv)))

#     a = torch.argmax(policy).item()
#     print(f"The expected move had a value of {[v for v, x, y, z, m in moves_vals if m == player_move][0]:.5f}")
#     # player_move = idx2move(board, (a//8)%8, a%8, a//64)
# #     arrows.append(svg.Arrow(player_move.from_square, player_move.to_square, color='#0f0'))
#     return svg.board(board=board, arrows=arrows, lastmove=player_move)
