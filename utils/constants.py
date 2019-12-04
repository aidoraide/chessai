from chess import Board, Move, ROOK, BISHOP, KNIGHT, QUEEN, PAWN, KING, WHITE, BLACK

DATA_DIR = 'data/'
ELO_CUTOFF = 1700
MAX_MOVES = 250
PROCESSED_FNAME = DATA_DIR + 'permove_data.csv'
PGN_PATH = DATA_DIR + 'lichess_db_standard_rated_2019-10.pgn/lichess_db_standard_rated_2019-10.pgn'
MODEL_PATH = 'models/chessnet.sd'

DIRECTIONS = ['E', 'NE', 'N', 'NW', 'W', 'SW', 'S', 'SE']
WHITE_DIRS, BLACK_DIRS = ['NW', 'N', 'NE'], ['SE', 'S', 'SW'] # Pawn directions
UNDER_PROMOS = [ROOK, BISHOP, KNIGHT]
WHITE_WIN = ' 1-0' # Note the space LOL. This took way too long to debug.
BLACK_WIN = '0-1'
TIE = '1/2-1/2'

LICHESS_CUTOFFS = {
    'top3%': 2204,
    'top2%': 2256,
    'top1%': 2342,
    'top0.3%': 2470
}
