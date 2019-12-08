from utils.neural_utils import get_net, board2tensor, move2idx, idx2move

from collections import defaultdict
from math import sqrt
from time import time
from heapq import heappush, heappop, heapify

import chess
import numpy as np
# np.random.seed(42)

class HashBoard(chess.Board):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._hash = hash(str(self))
    
    def __hash__(self):
        return self._hash

    def next_state(self, move):
        cp = super(HashBoard, self).copy()
        cp.push(move)
        cp._hash = hash(str(cp))
        return cp


class Chess:
    _result2reward = {
        '1-0': -1,
        '0-1': -1,
        '1/2-1/2': 0
    }

    @staticmethod
    def isOver(state):
        return state.result() in Chess._result2reward
    
    @staticmethod
    def reward(state):
        return Chess._result2reward[state.result()]

    @staticmethod
    def getLegalMoves(state):
        return state.legal_moves

    @staticmethod
    def nextState(state, action):
        move = action2move(state, action)
        return state.next_state(move)

    @staticmethod
    def initialState():
        return HashBoard()


class MCTS:
    C = 4 # Exploration vs exploitation param. High C encourages exploration.
    
    prediction_time = 0
    move_selection_time = 0
    next_state_time = 0
    increment_time = 0
    heapify_time = 0

    thinking_time = 0


    def __init__(self, root):
        self.root = root
        self.P = {}
        self.N = defaultdict(lambda: defaultdict(int))
        self.Q = defaultdict(lambda: defaultdict(int))
        self.U = {}
        self.Nsum = defaultdict(int)


    def pi(self, s, best_only=False):
        pi = np.zeros(73 * 8 * 8)

        if not best_only:
            for a, n in self.N[self.root].items():
                pi[a] = n
            return pi / pi.sum()
        else:
            max_n, best_a = 0, None
            for a, n in self.N[self.root].items():
                if n > max_n:
                    max_n, best_a = n, a
            pi[best_a] = 1
            return pi


    def get_u_val(self, Qsa, Psa, Nsums, Nsa):
        return Qsa + MCTS.C * Psa * sqrt(Nsums+1)/(1+Nsa)

    
    def search(self, s, game, nnet):
        if game.isOver(s):
            return -game.reward(s)

        Ps = self.P.get(s)

        if Ps is None:
            t0 = time()
            Ps, v = predict(nnet, s)
            MCTS.prediction_time += time() - t0

            t0 = time()
            Us = [ (-MCTS.C * Ps[a], a) # Q/N are 0 for initial U value
                for a in ((move2action(m) for m in game.getLegalMoves(s)))]
            heapify(Us)
            self.U[s], self.P[s] = Us, Ps
            MCTS.heapify_time += time() - t0
            return -v

        Qs, Ns, Us, Nsums = self.Q[s], self.N[s], self.U[s], self.Nsum[s]


        max_u, best_a, best_m = -float('inf'), None, None
        t0 = time()
        neg_u, best_a = heappop(Us)
        MCTS.move_selection_time += time() - t0

        t0 = time()
        sp = game.nextState(s, best_a)
        MCTS.next_state_time += time() - t0
        v = self.search(sp, game, nnet)

        # if s == self.root:
        #     print(f'Got v={v:.5f}           ')

        t0 = time()
        Qs[best_a] = (Ns[best_a]*Qs[best_a] + v)/(Ns[best_a] + 1)
        Ns[best_a] += 1
        self.Nsum[s] = Nsums + 1
        u = Qs[best_a] + MCTS.C * Ps[best_a]*sqrt(Nsums+1)/(1 + Ns[best_a])
        heappush(Us, (-u, best_a))
        MCTS.increment_time += time() - t0
        return -v


def predict(nnet, board):
    state = board2tensor(board).to('cuda:0')
    policy, value_out = nnet.forward(state.unsqueeze(0))
    # Remove batch size and return vector + scalar
    value = value_out[0].item()
    policy = policy[0].flatten().detach().cpu().numpy()
    return policy, value


def move2action(move):
    x, y, z = move2idx(move)
    return z * 64 + x * 8 + y


def action2move(state, a):
    x, y, z = (a//8)%8, a%8, (a//64)
    return idx2move(state, x, y, z)


nnet = get_net('models/oneshot1.sd')
board = HashBoard()
mcts = MCTS(board)
NUM_MCTS = 800
turn = 1
debug = True
while not Chess.isOver(board):
    print({chess.WHITE: 'White', chess.BLACK: 'Black'}[board.turn], f'turn #{turn}')
    print(board)
    mcts.root = board
    t0 = time()
    for i in range(NUM_MCTS):
        if i % 10 == 0:
            print(f'{i*100./NUM_MCTS:.1f}%  ({time() - t0:.2f}s)', end='\r')
        mcts.search(board, Chess, nnet)
    thinking_time = time() - t0
    MCTS.thinking_time += thinking_time
    pi = mcts.pi(board, best_only=turn > 30)
    action = np.random.choice(len(pi), p=pi)
    
    if debug:
        dbg = sorted([(v, a, action2move(board, a)) for a, v in enumerate(pi) if v != 0], reverse=True)
        for prob, a, move in dbg:
            print('   ', f'{str(move):8}', f'{prob:.4f}', '--> taking' if a == action else '')

        other_time = MCTS.thinking_time - MCTS.next_state_time - MCTS.prediction_time - MCTS.increment_time - MCTS.move_selection_time - MCTS.heapify_time
        print(f'MCTS.thinking_time       = {MCTS.thinking_time:.4f}    ({MCTS.thinking_time/turn:.2f}s per turn avg)')
        print(f'MCTS.next_state_time     = {MCTS.next_state_time:.4f}   ({MCTS.next_state_time/MCTS.thinking_time*100:.2f}%)')
        print(f'MCTS.prediction_time     = {MCTS.prediction_time:.4f}   ({MCTS.prediction_time/MCTS.thinking_time*100:.2f}%)')
        print(f'MCTS.increment_time      = {MCTS.increment_time:.4f}    ({MCTS.increment_time/MCTS.thinking_time*100:.2f}%)')
        print(f'MCTS.move_selection_time = {MCTS.move_selection_time:.4f}   ({MCTS.move_selection_time/MCTS.thinking_time*100:.2f}%)')
        print(f'MCTS.heapify_time        = {MCTS.heapify_time:.4f}   ({MCTS.heapify_time/MCTS.thinking_time*100:.2f}%)')
        print(f'MCTS.other_time          = {other_time:.4f}   ({other_time/MCTS.thinking_time*100:.2f}%)')
        print('Taking', action2move(board, action), f'({thinking_time:.2f}s)', '\n')

    board = Chess.nextState(board, action)
    turn += 1

print('\n', {chess.WHITE: 'White', chess.BLACK: 'Black'}[not board.turn], 'wins!')
print(board)
# while not Chess.isOver(board):
#     print(board)
#     P, v = predict(nnet, board)
#     Pp = np.zeros(73*8*8)
#     for a, m in [(move2action(m), m) for m in Chess.getLegalMoves(board)]:
#         Pp[a] = P[a]
#     move = action2move(board, np.argmax(Pp))
#     print('Taking', move)
#     board.push(move)

# print(board.result())
