"""
Monte Carlo AI for Chess
"""

import chess
import json
import os

from functools import reduce
from collections import defaultdict
from random import random
from time import time
from copy import deepcopy
from bisect import bisect_left, bisect_right
import numpy as np


INF = 1000000


initial_conf = {
    'piece2value': {
        chess.PAWN: 1,
        chess.KNIGHT: 3,
        chess.BISHOP: 4,
        chess.ROOK: 5,
        chess.QUEEN: 6,
        chess.KING: 0.1,
    },
    'VALUE_WEIGHT': 15,
    'CONTROL_WEIGHT': 0.5,
    'OD_WEIGHT': 2,
    'CHECK_WEIGHT': 5,
    'BACKED_MULT': 15,
    'PAWN_PROMO_WEIGHT': 1,
    'KING_PRESSURE_WEIGHT': 5,
}

def dump_conf(confname, conf):
    open('confs/' + confname + '.json', 'w').write(json.dumps(conf))


def load_conf(confname):
    conf = json.load(open('confs/' + confname + '.json'))

    # Convert int keys back to ints
    conf['piece2value'] = {int(k): v for k, v in conf['piece2value'].items()}
    return conf


def load_best_conf():
    conf_files = os.listdir('confs/')
    conf_files.remove('initial_conf.json')
    if not conf_files:
        return load_conf('initial_conf')
    return load_conf(conf_files[-1].split('.')[0])


dump_conf('initial_conf', initial_conf)


best_conf = load_best_conf()
MAX_TURNS = 50
def playout(board):
    starter_turn = board.turn
    history = simulate(board, best_conf, best_conf, MAX_TURNS)
    is_win = starter_turn != history[-1].turn and history[-1].result() in ('1-0', '0-1')
    is_tie = history[-1].result() in ('1/2-1/2', '*')
    return is_win, is_tie


class Board(chess.Board):
    def __init__(self, state=None):
        if state:
            super(Board, self).__init__(state)
        else:
            super(Board, self).__init__()
        self.state = str(self)


    def get_next_state(self, move):
        self.push(move)
        board = Board(self.fen())
        self.pop()
        return board


    def get_next_states(self):
        return [self.get_next_state(move) for move in self.legal_moves]


    def evaluate(self, conf=None, debug=False):
        if self.is_checkmate():
            return INF
        if self.is_insufficient_material() or self.is_seventyfive_moves() or self.is_stalemate():
            return -500

        if conf is None:
            conf = best_conf

        piece2value          = conf['piece2value']
        VALUE_WEIGHT         = conf['VALUE_WEIGHT']
        CONTROL_WEIGHT       = conf['CONTROL_WEIGHT']
        OD_WEIGHT            = conf['OD_WEIGHT']
        CHECK_WEIGHT         = conf['CHECK_WEIGHT']
        BACKED_MULT          = conf['BACKED_MULT']
        PAWN_PROMO_WEIGHT    = conf['PAWN_PROMO_WEIGHT']
        KING_PRESSURE_WEIGHT = conf['KING_PRESSURE_WEIGHT']


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
        our_pieces_value = sum((piece2value[p.piece_type] for i, p in our_pieces))
        opponent_pieces_value = sum((piece2value[p.piece_type] for i, p in opponent_pieces))
        for i, p in pieces:
            # Matters who is attacking who
            # Matters if the victim is backed up or not
            score = 0
            for ti, target in attacking[(i, p)]:                
                if len(backedBy[(ti, target)]) == 0:
                    # This piece can be considered sacrificed since it can be taken next turn.
                    # We treat it as a ratio of the pieces value to the other pieces we have.
                    # It is extra bad for our pieces to not be backed up because it is not our turn.
                    score = BACKED_MULT * (piece2value[p.piece_type] / piece2value[target.piece_type]) \
                        / (our_pieces_value / BACKED_MULT if target.color == us else opponent_pieces_value)
                else:
                    score = max(piece2value[p.piece_type] - piece2value[target.piece_type], 1) * 10
            
            score += len(backedBy[(i, p)]) * 16 / (len(our_pieces) if p.color == us else len(opponent_pieces))
            od_score += -score if p.color != us else score


        # Pawn promotion
        pawn_promo_score = 0


        # King safety
        our_king = [i for i, p in our_pieces if p.piece_type == chess.KING][0]
        opponent_king = [i for i, p in opponent_pieces if p.piece_type == chess.KING][0]

        our_king_pressure = len(self.attacks(our_king) & opponent_coverage)
        opponent_king_pressure = len(self.attacks(opponent_king) & our_coverage)

        REMAINING_POWER = 1
        # Weight the amount of squares the king is pressured on by the amount of remaining pieces. Few pieces remaining means king pressure is more important.
        king_pressure_score = opponent_king_pressure * ((16 - len(opponent_pieces)) / 15)**REMAINING_POWER \
            - our_king_pressure * ((16 - len(our_pieces) / 15))**REMAINING_POWER


        if debug:
            row_format = "{:>20}" * 4
            print(row_format.format('Variable', 'Weight', 'Score', 'Weighted Score'))
            rows = ['value', 'control', 'od', 'pawn_promo', 'king_pressure']
            print(row_format.format('value', VALUE_WEIGHT, value_score, value_score * VALUE_WEIGHT))
            print(row_format.format('control', CONTROL_WEIGHT, control_score, control_score * CONTROL_WEIGHT))
            print(row_format.format('od', OD_WEIGHT, od_score, od_score * OD_WEIGHT))
            print(row_format.format('pawn_promo', PAWN_PROMO_WEIGHT, pawn_promo_score, pawn_promo_score * PAWN_PROMO_WEIGHT))
            print(row_format.format('king_pressure', KING_PRESSURE_WEIGHT, king_pressure_score, king_pressure_score * KING_PRESSURE_WEIGHT))
            print(row_format.format('is_check', CHECK_WEIGHT, self.is_check(), self.is_check() * CHECK_WEIGHT))

        return value_score * VALUE_WEIGHT \
            + control_score * CONTROL_WEIGHT \
            + od_score * OD_WEIGHT \
            + pawn_promo_score * PAWN_PROMO_WEIGHT \
            + king_pressure_score * KING_PRESSURE_WEIGHT \
            + self.is_check() * CHECK_WEIGHT


    def __hash__(self):
        return self.state.__hash__()


def simulate(board, conf_white, conf_black, max_turns=INF):
    turns, history = 0, []
    while not board.is_game_over() and turns < max_turns:
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
        history.append(board)
        board = next_states[i]

    history.append(board)
    return history


def simulate_many(n, conf_white, conf_black, debug=False):
    games, whitewin, blackwin, tie, t0 = 0, 0, 0, 0, time()
    for _ in range(n):
        history = simulate(Board(), conf_white, conf_black)
        b = history[-1]

        games += 1
        whitewin += b.result() == '1-0'
        blackwin += b.result() == '0-1'
        tie += b.result() == '1/2-1/2'

        if debug:
            if b.is_fivefold_repetition():
                print("fivefold_repetition")
            if b.is_insufficient_material():
                print("insufficient_material")
            if b.is_seventyfive_moves():
                print("seventyfive_moves")
            if b.is_stalemate():
                print("stalemate")
            if b.is_variant_end():
                print("variant_end")
            
            row_format = "{:>15}" * 4
            print(len(history), 'turns')
            print("Average time per game", (time() - t0)/games)
            print(row_format.format('games', 'white', 'black', 'tie'))
            print(row_format.format(games, whitewin, blackwin, tie))
            TURNS = 5
            if b.result() == '1/2-1/2':
                for i, past_state in enumerate(history[-TURNS:]):
                    print('='*25 + str(-i) + '='*25)
                    evaluation = past_state.evaluate(conf_black if past_state.turn == chess.WHITE else conf_white, True)
                    print(-i, ' evaluates to ->', evaluation)
                    print(past_state)
                    print()
    
    return games, whitewin, blackwin, tie

