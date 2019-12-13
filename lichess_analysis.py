import os
import sys
import chess
import chess.pgn
import numpy as np
import pandas as pd
from collections import defaultdict
from utils import data_utils, constants

PGN_PATH = os.path.join('D:', 'lichess_db_standard_rated_2019-10.pgn/lichess_db_standard_rated_2019-10.pgn')

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

SIGNIFICANT_MOVES = 10000
def print_move_dist(board):
    move_map = {board.san(m):0 for m in board.legal_moves}
    values = np.array([0, 0, 0])
    target_state = board.fen()
    n_rows = 0
    print(f'target_state: {target_state}')
    for df in pd.read_csv(data_utils.LICHESS_DF_FNAME, chunksize=10000):
        n_rows += len(df)
        for row in df.itertuples(index=False):
            state, value, move = row.state, row.value, row.move
            if state == target_state:
                move_map[move] += 1
                values[int(value+1)] += 1

        moves = sum(move_map.values())
        if moves < SIGNIFICANT_MOVES:
            print(f'{moves*100/SIGNIFICANT_MOVES:5.2f}%', end='\r')
        else:
            print(f'This state represents {SIGNIFICANT_MOVES*100/n_rows:.2f}% of all move data')
            print('-'*45)
            print([f'{100*v:.2f}%' for v in values/values.sum()])
            for m in sorted(move_map.keys(), key=lambda m: -move_map[m]):
                c = move_map[m]
                print(f'{m:6}: {100*c/moves:8.2f}%')
            return

PIECES = [chess.PAWN, chess.ROOK, chess.KNIGHT, chess.BISHOP, chess.QUEEN, chess.KING]
def print_move_piece_type_dist():
    piece2string = {
        chess.PAWN: 'pawn',
        chess.ROOK: 'rook',
        chess.KNIGHT: 'knight',
        chess.BISHOP: 'bishop',
        chess.QUEEN: 'queen',
        chess.KING: 'king',
    }
    counts = {p:0 for p in PIECES}
    n_rows = 0
    for df in pd.read_csv(data_utils.LICHESS_DF_FNAME, chunksize=10000):
        n_rows += len(df)
        for row in df.itertuples(index=False):
            state, value, move = row.state, row.value, row.move
            board = chess.Board(state)
            m = board.parse_san(move)
            piece = board.piece_at(m.from_square)
            counts[piece.piece_type] += 1
        print(', '.join([f'{piece2string[p]:6}: {counts[p]*100/n_rows:6.2f}%' for p, c in counts.items()]), end='\r')

def print_move_from_square_dist():
    place2counts = list(range(64))
    n_rows = 0
    for df in pd.read_csv(data_utils.LICHESS_DF_FNAME, chunksize=10000):
        n_rows += len(df)
        for row in df.itertuples(index=False):
            state, value, move = row.state, row.value, row.move
            board = chess.Board(state)
            m = board.parse_san(move)
            place2counts[m.from_square] += 1
        if n_rows >= SIGNIFICANT_MOVES*10:
            break
    for p, c in enumerate(place2counts):
        x, y = p%8, p//8
        print(f'({x}, {y}): {c*100/n_rows:6.2f}%')


def print_move_dist_from_argv():
    board = chess.Board()
    for arg in sys.argv[1:]:
        board.push(board.parse_san(arg))
    print_move_dist(board)

if __name__ == '__main__':
    # print_best_possible_accuracy()
    print_move_dist_from_argv()
    # print_move_from_square_dist()
