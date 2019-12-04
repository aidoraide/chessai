import os
import chess.pgn
import numpy as np
from utils import data_utils

PGN_PATH = 'data/lichess_db_standard_rated_2019-10.pgn/lichess_db_standard_rated_2019-10.pgn'
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
