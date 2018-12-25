"""
Monte Carlo AI for Chess

NOTE:
For a Board, b

y in b.attacks(x) -> True if x can attack y's square (color of x and y doesn't matter). Could be an empty square too.

Possible connections:
x => y if x can kill y
x => y if x backs up y
x => y if x can be killed by y
x => y if x can is backed up by y

b.turn => who's turn it is
not b.turn => who just played

So if we are evaluating a future board, f, then:
not f.turn => our colour
f.turn => opponents colour

"""

import chess
import json

from functools import reduce
from collections import defaultdict
from random import random
from time import time
from bisect import bisect_left, bisect_right
import numpy as np


conf = {
    'piece2value': {
        chess.PAWN: 1,
        chess.KNIGHT: 3,
        chess.BISHOP: 4,
        chess.ROOK: 5,
        chess.QUEEN: 6,
        chess.KING: 10,
    },
    'VALUE_WEIGHT': 15,
    'CONTROL_WEIGHT': 1,
    'OD_WEIGHT': 2,
    'CHECK_WEIGHT': 5,
    'BACKED_MULT': 5,
    'PAWN_PROMO_WEIGHT': 1,
}

def dump_conf(fname, conf):
    open(fname, 'w').write(json.dumps(conf))


def load_conf(fname):
    conf = json.load(open(fname))

    # Convert int keys back to ints
    conf['piece2value'] = {int(k): v for k, v in conf['piece2value'].items()}
    return conf


class Board(chess.Board):
    def __init__(self, state=None):
        if state:
            super(Board, self).__init__(state)
        else:
            super(Board, self).__init__()
        self.state = str(self)


    def get_next_state(self, move):
        board = self.copy()
        board.push(move)
        return Board(board.fen())


    def evaluate(self, conf):
        if self.is_checkmate():
            return float('inf')

        piece2value          = conf['piece2value']
        VALUE_WEIGHT         = conf['VALUE_WEIGHT']
        CONTROL_WEIGHT       = conf['CONTROL_WEIGHT']
        OD_WEIGHT            = conf['OD_WEIGHT']
        CHECK_WEIGHT         = conf['CHECK_WEIGHT']
        BACKED_MULT          = conf['BACKED_MULT']
        PAWN_PROMO_WEIGHT    = conf['PAWN_PROMO_WEIGHT']


        us = not self.turn
        pieces = [(i, self.piece_at(i)) for i in range(64) if self.piece_at(i) is not None]
        our_pieces = [(i, p) for i, p in pieces if p.color == us]
        opponent_pieces = [(i, p) for i, p in pieces if p.color != us]

        # Value score measures the total value of your living pieces vs the value of your opponents
        value_score = sum((piece2value[p.piece_type] if p.color == us else -piece2value[p.piece_type] for i, p in pieces))

        # Board control score measures how much of the board is reachable/attackable by you
        """
        If we are controlling more squares then them, this is good.
        """
        our_coverage = reduce(lambda sqrset, ip: sqrset | self.attacks(ip[0]), our_pieces, chess.SquareSet())
        opponent_coverage = reduce(lambda sqrset, ip: sqrset | self.attacks(ip[0]), opponent_pieces, chess.SquareSet())
        control_score = len(our_coverage - opponent_coverage) - len(opponent_coverage - our_coverage)

        # Offense/Defence score measures how much your pieces are backed up and how much 
        # pressure they are applying to the opponent
        """
        If our piece is being attacked, and it is NOT being backed up, this is bad.
        If our piece is being backed up, and it is NOT being attacked, this is kinda good but probably not as important as ^.
        If their piece is being attacked. This is good.
        If their piece is being defended. This is bad.

        All of these can be multiplied by the weight of the piece being attacked - attacker


        For every piece, keep track of how many are attacking and how many are backing it up.

        For each piece p
            get p's attack set
            for each square in attack set
                if enemy in square, increment enemies attacked counter
                if ally in square, increment allys backed up counter


        4 maps piece => list(piece)
        attackedBy
        attacking
        backedBy
        backing
        """

        attackedBy, attacking, backedBy, backing = [defaultdict(list) for _ in range(4)]
        for i, p in pieces:
            for dest, targetp in [(dest, self.piece_at(dest)) for dest in self.attacks(i) if self.piece_at(dest) is not None]:
                if p.color == targetp.color:
                    backedBy[(dest, targetp)].append((i, p))
                    backing[(i, p)].append((dest, targetp))
                else:
                    attackedBy[(dest, targetp)].append((i, p))
                    attacking[(i, p)].append((dest, targetp))

        od_score = 0
        for i, p in pieces:
            # Matters who is attacking who
            # Matters if the victim is backed up or not
            score = 0
            for ti, target in attacking[(i, p)]:                
                if len(backedBy[(ti, target)]) == 0:
                    score = BACKED_MULT * piece2value[target.piece_type]
                else:
                    score = max(piece2value[p.piece_type] - piece2value[target.piece_type], 1)
            
            score += len(backedBy[(i, p)])
            od_score += -score if p.color != us else score


        # Pawn promotion
        pawn_promo_score = 0

        return value_score * VALUE_WEIGHT \
            + control_score * CONTROL_WEIGHT \
            + od_score * OD_WEIGHT \
            + pawn_promo_score * PAWN_PROMO_WEIGHT \
            + self.is_check() * CHECK_WEIGHT


    def __hash__(self):
        return self.state.__hash__()


def simulate(board, conf_white, conf_black):
    turns = 0
    while not board.is_game_over():
        turns += 1
        moves = list(board.legal_moves)
        next_states = [board.get_next_state(m) for m in moves]
        evals = [b.evaluate(conf_white if board.turn == chess.WHITE else conf_black) for b in next_states]
        base = min(evals)
        evals = [e - base for e in evals]
        moves, next_states, evals = list(zip(*sorted(zip(moves, next_states, evals), key=lambda x: x[2])))

        cutoff = bisect_left(evals, evals[-1] * 0.4)
        moves, next_states, evals = moves[cutoff:][::-1], next_states[cutoff:][::-1], evals[cutoff:][::-1]

        rand, total, i = random() * sum(evals), 0, -1
        while total < rand:
            i += 1
            total += evals[i]
        
        # print(evals)
        # print("taking evals", i, "->", evals[i], moves[i])
        # print(next_states[i])
        # print()
        board = next_states[i]

    return board, turns


def simulate_many(n, conf_white, conf_black):
    games, whitewin, blackwin, tie, t0 = 0, 0, 0, 0, time()
    for _ in range(n):
        b, turns = simulate(Board(), conf_white, conf_black)

        games += 1
        whitewin += b.result() == '1-0'
        blackwin += b.result() == '0-1'
        tie += b.result() == '1/2-1/2'

        # if b.is_fivefold_repetition():
        #     print("fivefold_repetition")
        # if b.is_insufficient_material():
        #     print("insufficient_material")
        # if b.is_seventyfive_moves():
        #     print("seventyfive_moves")
        # if b.is_stalemate():
        #     print("stalemate")
        # if b.is_variant_end():
        #     print("variant_end")
        
        # row_format = "{:>15}" * 4
        # print(turns, 'turns')
        # print("Average time per game", (time() - t0)/games)
        # print(row_format.format('games', 'white', 'black', 'tie'))
        # print(row_format.format(games, whitewin, blackwin, tie))
        # print(b)
        # print()
    
    return games, whitewin, blackwin, tie


import multiprocessing as mp


N = 100
THREADS = 10


def mutate_conf(conf, stddev=0.25):
    """
        Randomly tweaks conf with deltas from normal distribution
    """
    def r(d):
        for k, v in d.items():
            if isinstance(v, dict):
                r(v)
            elif isinstance(v, float) or isinstance(v, int):
                d[k] = v + np.random.normal(loc=0, scale=stddev)
    r(conf)
    return conf


def evolve_conf(initial_conf, gen=0):

    def worker(n, conf1, conf2, idx, out_q):
        games, whitewin, blackwin, tie = simulate_many(int(n/2), conf1, conf2)
        gamesR, whitewinR, blackwinR, tieR = simulate_many(int(n/2), conf2, conf1)
        out_q.put({'games': games+gamesR, 'conf1wins': whitewin + blackwinR, 'conf2wins': blackwin + whitewinR, 'ties': tie + tieR, 'index': idx})


    dump_conf('confs/gen_' + str(gen) + '.json', initial_conf)
    parent_conf = initial_conf
    while True:

        print("PARENT:", parent_conf)
        child_confs = [mutate_conf(load_conf('confs/gen_' + str(gen) + '.json')) for _ in range(THREADS)]

        out_q = mp.Queue()
        procs = [mp.Process(target=worker, args=(N, parent_conf, child_conf, idx, out_q)) for idx, child_conf in enumerate(child_confs)]
        for p in procs:
            p.start()

        print(THREADS, 'threads started')
        t0 = time()

        results = [out_q.get() for p in procs]
        for p in procs:
            p.join()
        results.sort(key=lambda res: res['index'])

        print('All procs finished in', time() - t0, 'seconds')

        best_child, best_winrate = 0, -1
        for i in range(THREADS):
            child_wins, parent_wins = results[i]['conf2wins'], results[i]['conf1wins']
            child_win_rate = child_wins / max(parent_wins, 1e-6)
            if child_win_rate > best_winrate:
                best_child = i
                best_winrate = child_win_rate

        if best_winrate > 1.25: # TODO find proper measure of statistical significance
            print("Updating parent to child", best_child, "with win rate:", best_winrate, '->', results[best_child])
            gen += 1
            parent_conf = child_confs[best_child]
            dump_conf('confs/gen_' + str(gen) + '.json', parent_conf)
        print()


start_gen = 19
conf = load_conf('confs/gen_' + str(start_gen) + '_conf.json')
evolve_conf(conf, start_gen)
