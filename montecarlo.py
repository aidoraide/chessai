"""
Monte Carlo AI for Chess

NOTE:
For a Board, b

y in b.attacks(x) -> True if x can attack y's square (color of x and y doesn't matter)

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
import numpy as np


piece2value = {
    chess.PAWN: 1,
    chess.ROOK: 5,
    chess.KNIGHT: 3,
    chess.BISHOP: 4,
    chess.QUEEN: 6,
    chess.KING: 10,
}
VALUE_WEIGHT = 10
CONTROL_WEIGHT = 1
OD_WEIGHT = 2

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


    def evaluate(self):
        us = not self.turn

        # Value score measures the total value of your living pieces vs the value of your opponents
        pieces = [self.piece_at(i) for i in range 64 if self.piece_at(i) is not None]
        value_score = sum((piece2value[p] if p.color == us else -piece2value[p] for p in pieces))

        # Board control score measures how much of the board is reachable/attackable by you
        board_score = len(self.legal_moves)

        # Offense/Defence score measures how much your pieces are backed up and how much 
        # pressure they are applying to the opponent
        # TODO: Build an attack graph? Use that somehow?

        return value_score * VALUE_WEIGHT + control_score * CONTROL_WEIGHT + od_score * OD_WEIGHT


    def __hash__(self):
        return self.state.__hash__()


b = Board()
b2n = {b: [b.get_next_state(m) for m in b.legal_moves]}
print(b)
print(b2n)
for nb in b2n[b]:
    print(nb)
    print()
