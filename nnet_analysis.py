import sys
import chess
import math
import numpy as np
import pandas as pd
from torch import nn
from model import load_model
from utils import data_utils
from utils.constants import device
from utils.neural_utils import board2tensor, get_legal_mask, move2idx

def print_value_policy(board, value_out, policy_out):
    value_out = value_out.view(1).item()
    policy_out = policy_out.view(73, 8, 8)
    move_map = {}
    for m in board.legal_moves:
        move = board.san(m)
        x, y, z = move2idx(m)
        move_map[move] = policy_out[z,x,y]
    print(f'value = {value_out:10.6f}')
    for m in sorted(move_map.keys(), key=lambda m: -move_map[m]):
        p = move_map[m]
        print(f'{m:6}: {p:8.2e} loss[y=1 {-math.log(p):8.4f}, y=0 {-math.log(1-p):8.4f}] grad[y=1 {-1/p:9.2e}, y=0 {1/(1-p):9.2e}]')



def show_nnet_output(board, move_str):
    data = data_utils.get_training_data(board.fen(), 0, move_str) # most common opener
    state, value, policy = data['state'].to(device), data['value'].to(device), data['policy'].to(device)

    nnet = load_model()
    policy_out, value_out = nnet(board2tensor(board).unsqueeze(0).to(device))
    policy_loss = nn.BCELoss(reduction="sum")(policy_out, policy).item()
    value_loss = nn.MSELoss(reduction="sum")(value_out, value).item()
    print(f'policy loss = {policy_loss}')
    print(f'value  loss = {value_loss}')
    policy_out = policy_out.view(73, 8, 8)
    policy_out *= get_legal_mask(board).to(device)
    policy_out /= policy_out.flatten().sum()

    print(f'{"-"*20} NNET OUT {"-"*20}')
    print_value_policy(board, value_out, policy_out)
    # print(f'{"-"*20} TYPICAL  {"-"*20}')
    # print_value_policy(board, value, policy)


if __name__ == '__main__':
    board = chess.Board()
    move_str = 'e4'
    for arg in sys.argv[1:-1]:
        board.push(board.parse_san(arg))
    if len(sys.argv) > 1:
        move_str = sys.argv[-1]
    show_nnet_output(board, move_str)
