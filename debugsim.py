from chesscontroller import load_best_conf, get_best_move, Board

conf = load_best_conf()
board = Board()
print(board)
while not board.is_game_over():
    board = get_best_move(board, conf, conf)
    _ = input()
    print(board)

print('\n', board.result())