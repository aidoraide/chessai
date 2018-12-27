from chesscontroller import load_conf, dump_conf, simulate_many

from time import time
from copy import deepcopy
from itertools import product
from random import shuffle

import multiprocessing as mp
import numpy as np


N = 500
THREADS = 10

if N % THREADS != 0:
    print('N (%d) must divide THREADS (%d)' % (N, THREADS))
    exit()

SHRINK = 0.85
GROW = 1.2


def mutate_conf(conf, stddev=0.0125):
    """
        Randomly tweaks conf with deltas from normal distribution
    """
    def r(d):
        for k, v in d.items():
            if isinstance(v, dict):
                r(v)
            elif isinstance(v, float) or isinstance(v, int):
                d[k] = v * (1 + np.random.normal(loc=0, scale=stddev))
    r(conf)
    return conf


def get_one_mutation_conf(conf, key, mult):
    conf = deepcopy(conf)
    def r(d):
        for k, v in d.items():
            if isinstance(v, dict):
                r(v)
            elif k == key:
                d[k] = v * mult
                break
    r(conf)
    return conf


def infinite_mutation_generator():
    conf = load_conf('initial_conf')
    keys = set()
    def r(d):
        for k, v in d.items():
            if isinstance(v, dict):
                r(v)
            elif isinstance(v, float) or isinstance(v, int):
                keys.add(k)
    r(conf)

    while True:
        mutations = [m for m in product([SHRINK, GROW], keys)]
        shuffle(mutations)
        for mult, key in mutations:
            yield mult, key


def worker(n, conf1, conf2, idx, out_q):
    games, whitewin, blackwin, tie = simulate_many(int(n/2), conf1, conf2)
    gamesR, whitewinR, blackwinR, tieR = simulate_many(int(n/2), conf2, conf1)
    out_q.put({'games': games+gamesR, 'conf1wins': whitewin + blackwinR, 'conf2wins': blackwin + whitewinR, 'ties': tie + tieR, 'index': idx})


def parallel_competition(parent_conf, child_conf):
    out_q = mp.Queue()
    procs = [mp.Process(target=worker, args=(N/THREADS, parent_conf, child_conf, idx, out_q)) for idx in range(THREADS)]
    for p in procs:
        p.start()

    print(THREADS, 'threads started')
    t0 = time()

    results = [out_q.get() for p in procs]
    for p in procs:
        p.join()

    child_wins = sum((r['conf2wins'] for r in results))
    parent_wins = sum((r['conf1wins'] for r in results))
    ties = sum((r['ties'] for r in results))
    games = sum((r['games'] for r in results))

    print('All procs finished in', time() - t0, 'seconds')

    return {'child_wins': child_wins, 'parent_wins': parent_wins, 'ties': ties, 'games': games}


def evolve_conf(initial_conf, gen=0):


    dump_conf('gen_' + str(gen), initial_conf)
    parent_conf = initial_conf
    for mult, key in infinite_mutation_generator():

        print("PARENT:", parent_conf)

        while True:
            print('Mutation ->', (mult, key))
            child_conf = get_one_mutation_conf(parent_conf, key, mult)
            result = parallel_competition(parent_conf, child_conf)
            child_wins, parent_wins, ties = result['child_wins'], result['parent_wins'], result['ties']
            child_winrate = child_wins / (parent_wins if parent_wins != 0 else 1e-6)

            # Estimate what is valid as statistically significant performance
            if child_winrate > 1.15 - min(0, 200 - ties)/1000 and ties < N - 150:
                print("Updating parent to child", (mult, key), "with win rate:", child_winrate, '->', result, '\n')
                gen += 1
                parent_conf = child_conf
                dump_conf('gen_' + str(gen), parent_conf)
                mult = mult - (mult - 1)*0.15 # Move mult closer to one
                # Keep training new children with similar mutations
            else:
                print('Child results were not good enough ->', result, '\n')
                break


start_gen = 14
conf = load_conf('gen_' + str(start_gen))
evolve_conf(conf, start_gen)
