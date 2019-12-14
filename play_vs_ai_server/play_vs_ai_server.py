from flask import Flask, render_template, redirect, request, jsonify
import urllib
import json
import chess
import chess.svg
import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir) 

from model import load_model
from neural_mcts import get_move_mcts, HashBoard

def get_chess_colour(colstr):
    if colstr == 'white':
        return chess.WHITE
    else:
        return chess.BLACK

def get_colour_str(chess_colour):
    if chess_colour == chess.WHITE:
        return 'white'
    return 'black'


nnet = load_model()
nnet.eval() 
app = Flask(__name__)
@app.route('/')
def index():
    return render_template("index.html")

@app.route('/start')
def start():
    colour = request.args['player_colour']
    board = chess.Board()
    urlParams = {'board': board.fen(), 'player_colour': colour}
    return redirect('/play?' + urllib.parse.urlencode(urlParams))

@app.route('/play')
def play():
    board = chess.Board(request.args['board'])
    player_colour = request.args['player_colour']
    last_move = request.args.get('last_move')
    last_move = chess.Move.from_uci(last_move) if last_move else ''
    selected_square = request.args.get('selected_square')
    chess_square = chess.SQUARE_NAMES.index(selected_square) if selected_square else None
    target_squares = []
    promo_targets = []

    if chess_square is not None and board.piece_at(chess_square) and board.piece_at(chess_square).color == get_chess_colour(player_colour):
        targets = []
        for m in board.legal_moves:
            if m.from_square == chess_square:
                targets.append(m.to_square)
                target_squares.append(chess.SQUARE_NAMES[m.to_square])
                if m.promotion:
                    promo_targets.append(chess.SQUARE_NAMES[m.to_square])
        squares = chess.SquareSet(targets)
    else:
        squares = None
    
    board_svg = chess.svg.board(board, squares=squares, lastmove=last_move)

    game_over_message = ''
    if board.is_game_over():
        if board.turn == get_chess_colour(player_colour):
            game_over_message = 'Game over, you lose'
        else:
            game_over_message = 'You win!'
    return render_template(
        "play.html",
        board_svg=board_svg,
        board_fen=board.fen(),
        target_squares=json.dumps(target_squares),
        promo_targets=json.dumps(promo_targets),
        selected_square=selected_square,
        player_colour=player_colour,
        turn=get_colour_str(board.turn),
        game_over_message=game_over_message,
        last_move=str(last_move)
    )

@app.route('/play/action')
def action():
    board = chess.Board(request.args['board'])
    player_colour = request.args['player_colour']
    from_square = request.args['from_square']
    to_square = request.args['to_square']
    promotion = request.args['promotion']

    move = board.parse_uci(f'{from_square}{to_square}{promotion}')
    board.push(move)

    urlParams = {'board': board.fen(), 'player_colour': player_colour, 'last_move': str(move)}
    return redirect('/play?' + urllib.parse.urlencode(urlParams))

@app.route('/get_ai_move')
def get_ai_move():
    board = HashBoard(request.args['board'])
    player_colour = request.args['player_colour']
    print('Get AI move for board', board.fen())
    move = get_move_mcts(nnet, board, evaluation_mode=True)
    move_str = board.uci(move)
    return jsonify({
        'from_square': move_str[:2],
        'to_square': move_str[2:4],
        'promotion': move_str[4:],
    })

if __name__ == '__main__':
    app.run(port=9999, host='0.0.0.0', debug=True)
