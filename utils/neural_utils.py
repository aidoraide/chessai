from chess import Board, Move, ROOK, BISHOP, KNIGHT, QUEEN, PAWN, KING, WHITE, BLACK
from collections import OrderedDict, Counter

import itertools

import torch
import torch.nn as nn
from utils.constants import device

N_POINTS = 1*8 + 3*2 + 3*2 + 5*2 + 9
piece2point = {
    PAWN: 1/N_POINTS,
    BISHOP: 3/N_POINTS,
    KNIGHT: 3/N_POINTS,
    ROOK: 5/N_POINTS,
    QUEEN: 9/N_POINTS,
    KING: 0,
}

#    +
#    N
# -W   E+
#    S
#    -
DIRECTIONS = ['E', 'NE', 'N', 'NW', 'W', 'SW', 'S', 'SE']
WHITE_DIRS, BLACK_DIRS = ['NW', 'N', 'NE'], ['SE', 'S', 'SW'] # Pawn directions
UNDER_PROMOS = [ROOK, BISHOP, KNIGHT]
def deltas2dir(dx, dy):
    if dx > 0 and dy == 0:
        return 'E'
    elif dx > 0 and dy > 0:
        return 'NE'
    elif dx == 0 and dy > 0:
        return 'N'
    elif dx < 0 and dy > 0:
        return 'NW'
    elif dx < 0 and dy == 0:
        return 'W'
    elif dx < 0 and dy < 0:
        return 'SW'
    elif dx == 0 and dy < 0:
        return 'S'
    elif dx > 0 and dy < 0:
        return 'SE'
    else:
        raise ValueError('dx and dy cannot both be 0')


# Since our NN will output an 73x8x8 policy vector we will use these to translate 
# between NN and Board interpretation of a move
def move2idx(move):
    x, y = move.from_square % 8, move.from_square // 8
    dx, dy = move.to_square % 8 - x, move.to_square // 8 - y
    distance, direction = max(abs(dx), abs(dy)), deltas2dir(dx, dy)
    if move.promotion in UNDER_PROMOS: # Pawn underpromotion
        assert y in (6, 1)
        z = UNDER_PROMOS.index(move.promotion) * 3 + 64 + \
            (WHITE_DIRS if y == 6 else BLACK_DIRS).index(direction)
    elif {abs(dx), abs(dy)} == {1, 2}: # Knight move
        z = (((abs(dx) == 2) << 2) | ((dx > 0) << 1) | (dy > 0)) + 56
    else: # Queen move or promo to Queen
        z = DIRECTIONS.index(direction) * 7 + distance - 1
    return x, y, z

xy2uci = lambda x, y: 'abcdefgh'[x] + str(y+1)

def idx2move(board, x, y, z):
    if z < 56: # Queen move or promo to Queen
        distance = z % 7 + 1
        direction = DIRECTIONS[z // 7]
        dx = distance * (-1 if 'W' in direction else 1 if 'E' in direction else 0)
        dy = distance * (-1 if 'S' in direction else 1 if 'N' in direction else 0)
        promo = 'q' if board.piece_at(x + y * 8).piece_type == PAWN and (y + dy in (0, 7)) else ''
    elif z < 64: # Knight move
        z -= 56
        dx = 2 if ((z >> 2) & 1) else 1
        dy = 2 if dx == 1 else 1
        dx *= 1 if ((z >> 1) & 1) else -1
        dy *= 1 if ((z >> 0) & 1) else -1
        promo = ''
    elif z < 73: # Pawn underpromotion
        assert y in (6, 1)
        z -= 64
        direction = (WHITE_DIRS if y == 6 else BLACK_DIRS)[z % 3]
        promo = UNDER_PROMOS[z // 3]
        promo = 'r' if promo == ROOK else 'n' if promo == KNIGHT else 'b'
        dx = -1 if 'W' in direction else 1 if 'E' in direction else 0
        dy = -1 if 'S' in direction else 1 if 'N' in direction else 0
    else: # WTF
        raise ValueError('Invalid z >= 73. z = %d.' % z)

    try:
        return Move.from_uci(xy2uci(x, y) + xy2uci(x+dx, y+dy) + promo)
    except IndexError as e:
        print('idx2move Index Error:', (x, y), 'd =', (dx, dy), promo)
        raise e


piece2plane = {val: idx for idx, val in enumerate([
    (WHITE, PAWN),
    (WHITE, ROOK),
    (WHITE, KNIGHT),
    (WHITE, BISHOP),
    (WHITE, QUEEN),
    (WHITE, KING),
    (BLACK, PAWN),
    (BLACK, ROOK),
    (BLACK, KNIGHT),
    (BLACK, BISHOP),
    (BLACK, QUEEN),
    (BLACK, KING),
    (WHITE, 'has_kingside_castling_rights'),
    (WHITE, 'has_queenside_castling_rights'),
    (BLACK, 'has_kingside_castling_rights'),
    (BLACK, 'has_queenside_castling_rights'),
    'turn'
])}

turn2sign = {
    BLACK: 1,
    WHITE: -1,
}

def get_where_state_white_mask(batch):
    return batch[:,piece2plane['turn'],0,0] == 1

def board2tensor(board):
    tensor = torch.zeros([len(piece2plane), 8, 8], dtype=torch.float32)

    for x, y in itertools.product(range(8), range(8)):
        sqr = x + y*8
        piece = board.piece_at(sqr)

        if not piece: continue
        # 12 Planes for each piece/color combination
        plane = piece2plane[(piece.color, piece.piece_type)]
        tensor[plane][x][y] = 1

    # 4 planes for castling rights
    combinations = itertools.product((WHITE, BLACK), (board.has_queenside_castling_rights, board.has_kingside_castling_rights))
    for color, func in combinations:
        if func(color):
            plane = piece2plane[(color, func.__name__)]
            tensor[plane,:,:] = 1

    tensor[piece2plane['turn'],:,:] = 1 if board.turn == WHITE else 0

    return tensor


def state2policy(board, best_move_idx):
    tensor = torch.zeros([73, 8, 8], dtype=torch.float32)
    x, y, z = best_move_idx
    tensor[z][x][y] = 1
    return tensor


def get_illegal_mask(board):
    tensor = torch.ones([73, 8, 8], dtype=torch.float32)
    for x, y, z in (move2idx(m) for m in board.legal_moves):
        tensor[z][x][y] = 0
    return tensor

def get_legal_mask(board):
    return 1 - get_illegal_mask(board)

def is_legal_mask(board, m):
    return 1

def is_into_check_mask(board, m):
    return 1 if board.is_into_check(m) else 0

def is_capture_mask(board, m):
    return 1 if board.is_capture(m) else 0

def get_masks(board, *maskFuncs):
    tensors = [torch.zeros([73, 8, 8], dtype=torch.float32) for i in maskFuncs]
    for m, (x, y, z) in ((m, move2idx(m)) for m in board.legal_moves):
        for i, maskFunc in enumerate(maskFuncs):
            tensors[i][z][x][y] = maskFunc(board, m)
    return tensors

class ResBlock(nn.Module):
    def __init__(self):
        super(ResBlock, self).__init__()
        self.net = nn.Sequential(OrderedDict([
            ('conv1', nn.Conv2d(256, 256, 3, padding=1)),
            ('batchnorm1', nn.BatchNorm2d(256)),
            ('relu1', nn.ReLU(inplace=True)),
            ('conv2', nn.Conv2d(256, 256, 3, padding=1)),
            ('batchnorm1', nn.BatchNorm2d(256)),
        ]))
        self.relu_out = nn.ReLU(inplace=True)
        
    def forward(self, x):
        conv_out = self.net(x)
        return self.relu_out(conv_out + x)

    
class Flatten(nn.Module):
    def forward(self, x):
        return x.flatten(1)
