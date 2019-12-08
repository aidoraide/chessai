from chess import Board, Move, ROOK, BISHOP, KNIGHT, QUEEN, PAWN, KING, WHITE, BLACK
from collections import OrderedDict, Counter

import itertools

import torch
import torch.nn as nn
device = torch.device("cuda:0")

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
    *[f'legal_{i}' for i in range(73)],
    *[f'into_check_{i}' for i in range(73)],
    *[f'is_capture_{i}' for i in range(73)],
    (WHITE, 'score'),
    (BLACK, 'score'),
    'score_diff',
    (WHITE, 'point_map'),
    (BLACK, 'point_map'),
    'signed_point_map',
    (WHITE, 'is_pinned_map'),
    (BLACK, 'is_pinned_map'),
    *[('attack_graph', i) for i in range(64)],
    *[('reverse_attack_graph', i) for i in range(64)],
    'is_check',
    (WHITE, 'turn'),
    (BLACK, 'turn'),
])}

turn2sign = {
    BLACK: 1,
    WHITE: -1,
}

def get_where_state_white_mask(batch):
    plane = piece2plane[(WHITE, 'turn')]
    return batch[:,plane,0,0] == 1

def board2tensor(board):
    tensor = torch.zeros([len(piece2plane), 8, 8], dtype=torch.float32)
    
    scores = {WHITE: 0, BLACK: 0}
    for x, y in itertools.product(range(8), range(8)):
        sqr = x + y*8
        piece = board.piece_at(sqr)
        for color in (BLACK, WHITE):
            for atkSqr in board.attackers(color, sqr):
                ax, ay = atkSqr//8, atkSqr%8
                tensor[piece2plane[('attack_graph', atkSqr)],x,y] = 1
                tensor[piece2plane[('reverse_attack_graph', sqr)],ax,ay] = 1

        if not piece: continue
        # 12 Planes for each piece/color combination
        plane = piece2plane[(piece.color, piece.piece_type)]
        tensor[plane][x][y] = 1

        points = piece2point[piece.piece_type]
        scores[piece.color] += points
        tensor[piece2plane[(piece.color, 'point_map')]][x][y] = points
        tensor[piece2plane['signed_point_map']][x][y] = points * turn2sign[board.turn]
        tensor[piece2plane[(WHITE, 'is_pinned_map')]][x][y] = board.is_pinned(WHITE, sqr)
        tensor[piece2plane[(BLACK, 'is_pinned_map')]][x][y] = board.is_pinned(BLACK, sqr)

    tensor[piece2plane[(WHITE, 'score')],:,:] = scores[WHITE]
    tensor[piece2plane[(BLACK, 'score')],:,:] = scores[BLACK]
    tensor[piece2plane['score_diff'],:,:] = (scores[BLACK] - scores[WHITE]) * turn2sign[board.turn]

    # 4 planes for castling rights
    combinations = itertools.product((WHITE, BLACK), (board.has_queenside_castling_rights, board.has_kingside_castling_rights))
    for color, func in combinations:
        if func(color):
            plane = piece2plane[(color, func.__name__)]
            tensor[plane,:,:] = 1

    legal, into_check, capture = get_masks(board, is_legal_mask, is_into_check_mask, is_capture_mask)

    p1, p2 = piece2plane['legal_0'], piece2plane['legal_72']
    tensor[p1:p2+1,:,:] = legal

    p1, p2 = piece2plane['into_check_0'], piece2plane['into_check_72']
    tensor[p1:p2+1,:,:] = into_check

    p1, p2 = piece2plane['is_capture_0'], piece2plane['is_capture_72']
    tensor[p1:p2+1,:,:] = capture

    plane = piece2plane[(board.turn, 'turn')]
    tensor[plane,:,:] = 1

    tensor[piece2plane['is_check'],:,:] = 1 if board.is_check() else 0

    return tensor


def state2policy(board, best_move_idx):
    tensor = torch.zeros([73, 8, 8], dtype=torch.float32)
#     for x, y, z in (move2idx(m) for m in board.legal_moves):
#         tensor[z][x][y] = 1

    x, y, z = best_move_idx
    tensor[z][x][y] = 1               # Make the best move worth more
    return tensor / torch.sum(tensor) # Turn tensor into probabilities (sum(tensor) == 1)

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

    
class ChessNet(nn.Module):
    # def __init__(self, n_res_blocks=19, learning_rate=0.01, bias=False, gpu_id=0):
    #     super(ChessNet, self).__init__()
    #     res_blocks = [(f'res_block{i+1}', ResBlock()) for i in range(n_res_blocks)]
    #     self.res_tower = nn.Sequential(OrderedDict([
    #         ('conv1', nn.Conv2d(len(piece2plane), 256, 3, padding=1)),
    #         ('batchnorm1', nn.BatchNorm2d(256)),
    #         ('relu1', nn.ReLU(inplace=True)),
    #         *res_blocks
    #     ]))
    #     self.policy_head = nn.Sequential(OrderedDict([
    #         ('conv1', nn.Conv2d(256, 2, 1)),
    #         ('batchnorm1', nn.BatchNorm2d(2)),
    #         ('relu1', nn.ReLU(inplace=True)),
    #         ('flatten', Flatten()),
    #         ('fc1', nn.Linear(8*8*2, 8*8*73)),
    #         ('softmax', nn.Softmax(dim=1))
    #     ]))
    #     self.value_head = nn.Sequential(OrderedDict([
    #         ('conv1', nn.Conv2d(256, 1, 1)),
    #         ('batchnorm1', nn.BatchNorm2d(1)),
    #         ('relu1', nn.ReLU(inplace=True)),
    #         ('flatten', Flatten()), # Should we flatten here or after? Does it matter?
    #         ('fc1', nn.Linear(64, 256)),
    #         ('relu2', nn.ReLU(inplace=True)),
    #         ('fc2', nn.Linear(256, 1)),
    #         ('tanh', nn.Tanh())
    #     ]))
    def __init__(self, n_res_blocks=19, learning_rate=0.01, bias=False, gpu_id=0):
        super(ChessNet, self).__init__()
        res_blocks = [(f'res_block{i+1}', ResBlock()) for i in range(n_res_blocks)]
        self.res_tower = nn.Sequential(OrderedDict([
            ('conv1', nn.Conv2d(len(piece2plane), 256, 3, padding=1)),
            ('batchnorm1', nn.BatchNorm2d(256)),
            ('relu1', nn.ReLU(inplace=True)),
            *res_blocks
        ]))
        self.policy_head = nn.Sequential(OrderedDict([
            ('conv1', nn.Conv2d(256, 64, 1)), # Changed to output 64 filters instead of 2
            ('batchnorm1', nn.BatchNorm2d(64)),
            ('relu1', nn.ReLU(inplace=True)),
            ('conv2', nn.Conv2d(64, 16, 1)), # Added this conv layer
            ('batchnorm2', nn.BatchNorm2d(16)),
            ('relu2', nn.ReLU(inplace=True)),
            ('flatten', Flatten()),
            ('fc1', nn.Linear(8*8*16, 8*8*73)),
            ('softmax', nn.Softmax(dim=1))
        ]))
        self.value_head = nn.Sequential(OrderedDict([
            ('conv1', nn.Conv2d(256, 32, 1)), # Changed to 32 out instead of 2
            ('batchnorm1', nn.BatchNorm2d(32)),
            ('relu1', nn.ReLU(inplace=True)),
            ('conv2', nn.Conv2d(32, 4, 1)), # Changed to 32 out instead of 2
            ('batchnorm2', nn.BatchNorm2d(4)),
            ('relu2', nn.ReLU(inplace=True)),
            ('flatten', Flatten()),
            ('fc1', nn.Linear(64 * 4, 512)),
            ('relu2', nn.ReLU(inplace=True)),
            ('fc2', nn.Linear(512, 1)),
            ('tanh', nn.Tanh())
        ]))
    # def __init__(self, n_res_blocks=19, learning_rate=0.01, bias=False, gpu_id=0):
    #     super(ChessNet, self).__init__()
    #     res_blocks = [(f'res_block{i+1}', ResBlock()) for i in range(n_res_blocks)]
    #     self.res_tower = nn.Sequential(OrderedDict([
    #         ('conv1', nn.Conv2d(len(piece2plane), 256, 3, padding=1)),
    #         ('batchnorm1', nn.BatchNorm2d(256)),
    #         ('relu1', nn.ReLU(inplace=True)),
    #         *res_blocks
    #     ]))
    #     self.policy_head = nn.Sequential(OrderedDict([
    #         ('conv1', nn.Conv2d(256, 64, 1)), # Changed to output 64 filters instead of 2
    #         ('batchnorm1', nn.BatchNorm2d(64)),
    #         ('relu1', nn.ReLU(inplace=True)),
    #         ('conv2', nn.Conv2d(64, 16, 1)), # Added this conv layer
    #         ('batchnorm2', nn.BatchNorm2d(16)),
    #         ('relu2', nn.ReLU(inplace=True)),
    #         ('flatten', Flatten()),
    #         ('fc1', nn.Linear(8*8*16, 8*8*73)),
    #         ('softmax', nn.Softmax(dim=1))
    #     ]))
    #     self.value_head = nn.Sequential(OrderedDict([
    #         ('conv1', nn.Conv2d(256, 2, 1)), # Changed to 32 out instead of 2
    #         ('batchnorm1', nn.BatchNorm2d(2)),
    #         ('relu1', nn.ReLU(inplace=True)),
    #         ('flatten', Flatten()),
    #         ('fc1', nn.Linear(64 * 2, 512)),
    #         ('relu2', nn.ReLU(inplace=True)),
    #         ('fc2', nn.Linear(512, 1)),
    #         ('tanh', nn.Tanh())
    #     ]))


    def forward(self, x):
        tower_out = self.res_tower(x)
        policy_out = self.policy_head(tower_out)
        value_out = self.value_head(tower_out)
        
        return policy_out.view(-1, 73, 8, 8), value_out.view(-1, 1)


def get_net(fname):
    nnet = ChessNet(n_res_blocks=19)
    nnet.eval()
    nnet.to(device)
    nnet.load_state_dict(torch.load(fname))
    return nnet


def predict(nnet, board):
    state = board2tensor(board)
    policy, value_out = nnet.forward(state)
