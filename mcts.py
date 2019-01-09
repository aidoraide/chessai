from chesscontroller import playout, get_result, get_player

from bisect import bisect_left
from math import log, sqrt
from random import choice
from time import time

import numpy as np
import multiprocessing as mp


CUTOFF = 0.75        # Moves must be at least CUTOFF times as good as the best move to be considered
C = 0.10             # Constant for UCB1
AFFINE_SD = 0.1      # Standard deviation after affine transform
SECONDS_TO_CALC = 30 # Time we allow MCTS to search the tree
THREADS = 10         # Number of threads for parallel processing
HEURISTIC_WEIGHT = 1 # Amount of playouts the heuristic evaluation counts for when calculating confidence intervals


class MCTNode:
    P1 = 1
    P2 = 2
    TIE = 0


    def __init__(self, state, player, heuristic_evaluation=0):
        self.state = state
        self.wins = 0
        self.plays = 0
        self.ties = 0
        self.player = player
        self.heuristic_evaluation = heuristic_evaluation
        self.children = []


    def add_record(self, winner):        
        self.plays += 1
        self.wins += winner == self.player
        self.ties += winner == MCTNode.TIE


    def __hash__(self):
        return self.state.__hash__()


def get_best_child(mctnode, use_ci=True, debug_nodes=None):
    if mctnode.plays == 0:
        raise ValueError('Node must have more than 0 plays')
    if any((c.plays == 0 for c in mctnode.children)):
        raise ValueError('Child nodes must have more than 0 plays')

    if debug_nodes and mctnode in debug_nodes:
        means = [(c.heuristic_evaluation * HEURISTIC_WEIGHT + c.wins/2 + - (c.plays - c.ties - c.wins)/2)
                / (c.plays + HEURISTIC_WEIGHT) for c in mctnode.children]
        interval = [sqrt(C * log(mctnode.plays) / c.plays) for c in mctnode.children]
        idxs = sorted(list(range(len(means))), key=lambda i: means[i] + interval[i])
        row_format = "{:>10}" + "{:>15}"*3 + '{:>5}'*3
        print()
        print(row_format.format('Index', 'Mean', '1/2 Interval', 'Upper Bound', 'W', 'L', 'T'))
        for i in idxs:
            child = mctnode.children[i]
            w, l, t = child.wins, child.plays - child.wins - child.ties, child.ties
            print(("{:>10}" + "{:>15.4f}"*3 + '{:>5}'*3).format(i, means[i], interval[i], means[i] + interval[i], w, l, t))
    
    best_child = np.argmax([
        # Mean payout for child. Weight the heuristic to vary how important it is in determining payout.
        (c.heuristic_evaluation * HEURISTIC_WEIGHT + c.wins/2 + - (c.plays - c.ties - c.wins)/2)
        / (c.plays + HEURISTIC_WEIGHT)
        
        # Upper bound of CI
        + (sqrt(C * log(mctnode.plays) / c.plays) if use_ci else 0)
        for c in mctnode.children])
    return mctnode.children[best_child]


def playout_parallel(states):
    def worker(state, idx, out_q):
        winner = playout(state, MCTNode.P1, MCTNode.P2, MCTNode.TIE)
        out_q.put({'idx': idx, 'winner': winner})

    out_q = mp.Queue()
    procs = [mp.Process(target=worker, args=(state, idx, out_q)) for idx, state in enumerate(states)]
    for p in procs:
        p.start()

    results = [None for _ in states]
    for _ in states:
        res = out_q.get()
        results[res['idx']] = res['winner']
    
    for p in procs:
        p.join()

    return results


def get_move(state):

    root = MCTNode(state, get_player(state, MCTNode.P1, MCTNode.P2))
    
    def mcts(node):
        if node.children:
            # Have all children and statistics for each of them so recurse
            results = mcts(get_best_child(node, debug_nodes={root}))
            for winner in results:
                node.add_record(winner)
            return results

        next_states = node.state.get_next_states()
        if not next_states:
            # Game over state
            winner = get_result(node.state, MCTNode.P1, MCTNode.P2, MCTNode.TIE)
            node.add_record(winner)
            return [winner]

        # By here, we dont have any children on this node, so we must populate them by making one child per available thread
        # and doing a playout for each of those children in parallel. Use heuristics to trim the ones that don't look good.

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

        child_player = MCTNode.P1 if node.player == MCTNode.P2 else MCTNode.P2
        for child, evaluation in zip(next_states, evals):
            node.children.append(MCTNode(child, child_player, evaluation))

        results = playout_parallel([c.state for c in node.children])
        for child, result in zip(node.children, results):
            child.add_record(result)
        for winner in results:
            node.add_record(winner)
        return results


    
    t0 = time()
    while time() - t0 < SECONDS_TO_CALC:
        mcts(root)
    best_child = get_best_child(root, use_ci=False, debug_nodes={root})
    print('Picked child', root.children.index(best_child))
    return best_child.state
