from utils.neural_utils import board2tensor, move2idx, idx2move, get_legal_mask
from utils.constants import device
from model import load_model

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

    def col(self):
        return "white" if self.turn == chess.WHITE else "black"

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
    C = 1 # Exploration vs exploitation param. High C encourages exploration.
    
    prediction_time = 0
    move_selection_time = 0
    next_state_time = 0
    increment_time = 0
    heapify_time = 0

    thinking_time = 0

    pred_preprocessing_time = 0
    pred_forward_time = 0
    pred_postprocessing_time = 0


    def __init__(self, root):
        self.root = root
        self.P = {}
        self.N = defaultdict(lambda: defaultdict(int))
        self.Q = defaultdict(lambda: defaultdict(int))
        self.U = {}
        self.Nsum = defaultdict(int)


    def pi(self, s, best_only=False):
        pi = np.zeros(73 * 8 * 8)

        for a, n in self.N[self.root].items():
            print(f'{action2move(self.root, a)}: p={self.P[self.root][a]:.5f} N={n}')

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
        # if s.board_fen() == HashBoard().board_fen():
        #     print('-'*60)
        #     for m in game.getLegalMoves(s):
        #         a = move2action(m)
        #         q = Qs[a]
        #         p = Ps[a]
        #         b = MCTS.C * Ps[a]*sqrt(Nsums+1)/(1 + Ns[a])
        #         print(f'{m}: Q={q:9.2e}, P={p:9.2e}, B={b:9.2e}, Q/B={abs(q/b):8.4f}, Nsums={Nsums:8}, Ns={Ns[a]:8}, R={sqrt(Nsums+1)/(1 + Ns[a]):8.4f}')
        heappush(Us, (-u, best_a))
        MCTS.increment_time += time() - t0
        return -v


def predict(nnet, board):
    t0 = time()
    state = board2tensor(board).to(device)
    MCTS.pred_preprocessing_time += time() - t0

    t0 = time()
    policy, value_out = nnet.forward(state.unsqueeze(0))
    # Remove batch size and return vector + scalar
    value = value_out[0].item()
    policy = policy[0].flatten().detach().cpu().numpy()
    MCTS.pred_forward_time += time() - t0

    t0 = time()
    noise = 0.3 # np.random.dirichlet([0.03], 1).flatten()
    n = 0.25
    # print('mean =', noise.mean(), 'std =', noise.std())
    # policy = (1-n)*policy + n*noise
    policy *= get_legal_mask(board).flatten().detach().numpy()
    policy = policy/policy.sum()
    MCTS.pred_postprocessing_time += time() - t0
    return policy, value


def move2action(move):
    x, y, z = move2idx(move)
    return z * 64 + x * 8 + y


def action2move(state, a):
    x, y, z = (a//8)%8, a%8, (a//64)
    return idx2move(state, x, y, z)


NUM_MCTS = 800
debug = True
def get_move_mcts(nnet, board, evaluation_mode=False):
    
    t0 = time()
    mcts = MCTS(board)
    mcts.root = board
    for i in range(NUM_MCTS+1):
        if i % 25 == 0:
            print(f'{i*100./NUM_MCTS:.1f}%  ({time() - t0:.2f}s)', end='\r')
        mcts.search(board, Chess, nnet)
    thinking_time = time() - t0
    MCTS.thinking_time += thinking_time
    pi = mcts.pi(board, best_only=evaluation_mode)
    action = np.random.choice(len(pi), p=pi)
    if debug:
        dbg = sorted([(v, a, action2move(board, a)) for a, v in enumerate(pi) if v != 0], reverse=True)
        for prob, a, move in dbg:
            print('   ', f'{board.san(move):8}', f'{prob:.4f}({int(round(prob*NUM_MCTS))})', '--> taking' if a == action else '')

        other_time = MCTS.thinking_time - MCTS.next_state_time - MCTS.prediction_time - MCTS.increment_time - MCTS.move_selection_time - MCTS.heapify_time
        # print(f'MCTS.thinking_time       = {MCTS.thinking_time:.4f}    ({MCTS.thinking_time/turn:.2f}s per turn avg)')
        print(f'MCTS.next_state_time     = {MCTS.next_state_time:.4f}   ({MCTS.next_state_time/MCTS.thinking_time*100:.2f}%)')
        print(f'MCTS.prediction_time     = {MCTS.prediction_time:.4f}   ({MCTS.prediction_time/MCTS.thinking_time*100:.2f}%)')
        print(f'    MCTS.preprocessing_time      = {MCTS.pred_preprocessing_time:.4f}   ({MCTS.pred_preprocessing_time/MCTS.prediction_time*100:.2f}%)')
        print(f'    MCTS.forward_time            = {MCTS.pred_forward_time:.4f}   ({MCTS.pred_forward_time/MCTS.prediction_time*100:.2f}%)')
        print(f'    MCTS.postprocessing_time     = {MCTS.pred_postprocessing_time:.4f}   ({MCTS.pred_postprocessing_time/MCTS.prediction_time*100:.2f}%)')
        print(f'MCTS.increment_time      = {MCTS.increment_time:.4f}    ({MCTS.increment_time/MCTS.thinking_time*100:.2f}%)')
        print(f'MCTS.move_selection_time = {MCTS.move_selection_time:.4f}   ({MCTS.move_selection_time/MCTS.thinking_time*100:.2f}%)')
        print(f'MCTS.heapify_time        = {MCTS.heapify_time:.4f}   ({MCTS.heapify_time/MCTS.thinking_time*100:.2f}%)')
        print(f'MCTS.other_time          = {other_time:.4f}   ({other_time/MCTS.thinking_time*100:.2f}%)')
        print('Taking', action2move(board, action), f'({thinking_time:.2f}s)', '\n')
    move = action2move(board, action)
    return move

if __name__ == '__main__':
    nnet = load_model()
    board = HashBoard()
    mcts = MCTS(board)
    turn = 1
    while not Chess.isOver(board):
        print({chess.WHITE: 'White', chess.BLACK: 'Black'}[board.turn], f'turn #{turn}')
        print(board)
        mcts.root = board
        t0 = time()
        for i in range(NUM_MCTS+1):
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
                print('   ', f'{str(move):8}', f'{prob:.4f}({int(round(prob*NUM_MCTS))})', '--> taking' if a == action else '')

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
