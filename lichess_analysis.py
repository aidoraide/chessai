import os
import chess.pgn
import numpy as np
import pandas as pd
from collections import defaultdict
from utils import data_utils

PGN_PATH = 'data/lichess_db_standard_rated_2019-10.pgn/lichess_db_standard_rated_2019-10.pgn'

def print_elo_dist():
    pgn = open(PGN_PATH)

    MAX_ELO = 10000
    elos = np.zeros((MAX_ELO+1))

    headers = chess.pgn.read_headers(pgn)
    while headers is not None:
        white_elo, black_elo = int(headers['WhiteElo']), int(headers['BlackElo'])
        elos[min(white_elo, black_elo, MAX_ELO)] += 1
        headers = chess.pgn.read_headers(pgn)

    lteElo = 0
    nGames = sum(elos)
    for elo in range(MAX_ELO+1):
        lteElo += elos[elo]
        print(f"{elo}: {lteElo} {lteElo/nGames*100:.1f}%")
        if lteElo == nGames:
            break

def print_best_possible_accuracy():
    # Prints ~91% for value, ~90.5% for policy
    n_rows = 0
    state2output_dist = defaultdict(lambda: {
        'value': [0, 0, 0], # l/t/w
        'moves': defaultdict(int)
    })
    for df in pd.read_csv(data_utils.LICHESS_DF_FNAME, chunksize=10000):
        n_rows += df.shape[0]
        print(f'Building dists: {n_rows/96395004*100:7.2f}%', end='\r')
        for row in df.itertuples(index=False):
            state, value, move = row.state, row.value, row.move
            state2output_dist[state]['value'][value+1] += 1
            state2output_dist[state]['moves'][move] += 1

    possible_policy_correct = 0
    possible_value_correct = 0
    for i, (state, dist) in enumerate(state2output_dist.items()):
        if i % 10000 == 0:
            print(f'Counting dists: {100*(i+1)/n_rows:7.2f}%', end='\r')
        value_dist, moves_dist = dist['value'], dist['moves']
        possible_value_correct += max(value_dist)
        possible_policy_correct += max(moves_dist.values())

    print(f'The best possible accuracies are:')
    print(f'    policy = {100*possible_policy_correct/n_rows:.2f}%')
    print(f'    value  = {100*possible_value_correct/n_rows:.2f}%')

    return (possible_policy_correct/n_rows, possible_value_correct/n_rows)

if __name__ == '__main__':
    print_best_possible_accuracy()
