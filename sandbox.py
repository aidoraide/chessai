import chesscontroller
from mcts import get_move

b = chesscontroller.Board()
while b.result() == '*':
    print(b)
    print()
    b = get_move(b)

print(b)