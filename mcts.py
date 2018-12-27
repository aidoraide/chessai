from chesscontroller import playout

from bisect import bisect_left
from math import log, sqrt
from random import choice
from time import time

import numpy as np


CUTOFF = 0.5         # Moves must be at least CUTOFF times as good as the best move to be considered
C = 2                # Constant for UCB1
AFFINE_SD = 0.2      # Standard deviation after affine transform
SECONDS_TO_CALC = 12 # Time we allow MCTS to search the tree
THREADS = 10         # Number of threads for parallel processing


class MCTNode:
    def __init__(self, state, heuristic_evaluation=0):
        self.state = state
        self.wins = 0
        self.plays = 0
        self.ties = 0
        self.heuristic_evaluation = heuristic_evaluation
        self.children = []


    def add_record(self, is_win, is_tie):
        if is_tie and is_win:
            raise ValueError('Cannot have a tie and win at the same time.')
        
        self.plays += 1
        self.wins += is_win
        self.ties += is_tie


    def __hash__(self):
        return self.state.__hash__()

def get_best_child(mctnode):
    if mctnode.plays == 0:
        raise ValueError('Node must have more than 0 plays')
    if any((c.plays == 0 for c in mctnode.children)):
        raise ValueError('Child nodes must have more than 0 plays')
    best_child = np.argmax([c.heuristic_evaluation + sqrt(C * log(mctnode.plays) / c.plays) for c in mctnode.children])
    return mctnode.children[best_child]


def get_move(state):

    
    def mcts(node):
        if not node.children:
            next_states = node.state.get_next_states()
            evals = [s.evaluate() for s in next_states]
            # Shift the evals vector up/down so the min is 0. This way we can choose only evaluations which are close to the best.
            evals_min = min(evals)
            evals = [x - evals_min for x in evals]
            next_states, evals = list(zip(*sorted(zip(next_states, evals), key=lambda x: x[1])))
            cutoff = max(bisect_left(evals, evals[-1] * CUTOFF), len(evals) - THREADS)
            next_states, evals = next_states[cutoff:], np.array(evals[cutoff:])

             # Affine transform our data so it is somewhat normalized
            std = np.std(evals)
            evals = evals * 0 if std == 0.0 else (evals - np.mean(evals)) * (AFFINE_SD / std)
            # At this point most of our evals are in the range [-0.35, 0.35]

            for child, evaluation in zip(next_states, evals):
                node.children.append(MCTNode(child, evaluation))
            unvisited = node.children
            print('created', len(node.children), 'children for a node', evals)
        else:
            unvisited = [c for c in node.children if c.plays == 0]
            if not unvisited:
                is_win, is_tie = mcts(get_best_child(node))
                node.add_record(is_win, is_tie)
                return is_win, is_tie

        to_visit = choice(unvisited)
        is_win, is_tie = playout(to_visit.state)
        to_visit.add_record(is_win, is_tie)
        node.add_record(is_win, is_tie)
        return is_win, is_tie


    root = MCTNode(state)
    t0 = time()
    while time() - t0 < SECONDS_TO_CALC:
        mcts(root)
    print('getting result', [(c.plays, c.wins, c.ties) for c in root.children])
    return get_best_child(root).state
