import chess
import pandas as pd
import numpy as np
import re
import os
import sys
import math
import json
import collections
from chess import Board, Move, ROOK, BISHOP, KNIGHT, QUEEN, PAWN, KING, WHITE, BLACK
from multiprocessing.pool import Pool
import torch
from torch.utils.data import Dataset, DataLoader
from torch import multiprocessing

from .constants import DATA_DIR, PROCESSED_FNAME, ELO_CUTOFF, MAX_MOVES, PGN_PATH, LICHESS_CUTOFFS
from .neural_utils import idx2move, move2idx, get_illegal_mask, board2tensor, state2policy

VALUE_DIST_DF_FNAME = PROCESSED_FNAME.split('.')[0] + '_value_dist.csv'
LICHESS_DF_FNAME = PROCESSED_FNAME.split('.')[0] + '_lichess.csv'
LICHESS_BROKEN_DF_FOLDER = LICHESS_DF_FNAME.split('.')[0]


def get_num_workers_from_args():
    if len(sys.argv) == 1:
        return 10
    return int(sys.argv[1])


class SupervisedChessDataset(Dataset):

    def get_raw(self, idx):
        row = self.df.iloc[idx, :]
        state, value, move = row.state, row.value, row.move
        board = Board(state)
        move = board.parse_san(move)
        return {
            'board': board,
            'value': value,
            'move': move,
        }

    def __init__(self, df, idxs):
        self.df = df.iloc[idxs]
        self.length = self.df.shape[0]
        print(__name__, 'dataset of length', self.length)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        row = self.df.iloc[idx, :]
        state, value, move = row.state, row.value, row.move
        board = Board(state)
        move_idxs = move2idx(board.parse_san(move))
        # attack_graph = get_attack_graph(board)
        return {
            # torch.cat((board2tensor(board), attack_graph), 0),
            'state': board2tensor(board),
            'value': torch.Tensor([value]),
            'policy': state2policy(board, move_idxs),
        }


def game2permove(d):
    result, moves = d
    data = []
    result = result.strip()
    moves = sum([m.strip().split()
                 for m in re.split(r'[0-9]+\. ', moves) if m], [])
    # value of state from player's perspective
    # 1 from winner's perspective, -1 from loser's perspective, 0 if tie
    value = 1 if result == '1-0' else -1 if result == '0-1' else 0
    board = Board()
    for move_str in moves:
        move = board.parse_san(move_str)

        assert idx2move(board, *move2idx(move)) == move

        data.append((board.fen(), value, move_str))

        # Move to next data point
        board.push(move)

        value *= -1

    return data


def generate_dataset():
    cvc_df = pd.read_csv(DATA_DIR + '2016_CvC.csv')
    cvh_df = pd.read_csv(DATA_DIR + '2016_CvH.csv')
    df = pd.concat([cvc_df, cvh_df])
    df.columns = [c.lower().replace(' ', '_') for c in df.columns]

    print(df.shape)
    print(df.head())

    df = df[(df['commentaries'].str.contains('disconnected') == False) &
            (df['white_elo'] > ELO_CUTOFF) &
            (df['black_elo'] > ELO_CUTOFF) &
            (df['plycount'] <= MAX_MOVES)]

    b = Board()
    #    +
    #    N
    # -W   E+
    #    S
    #    -

    state_data, value_data, move_data = [], [], []
    with Pool() as pool:
        games = pool.map(game2permove, zip(df.result, df.moves), chunksize=32)
        for gamedata in games:
            for sd, vd, md in gamedata:
                state_data.append(sd)
                value_data.append(vd)
                move_data.append(md)

    permove_df = pd.DataFrame(
        data={'state': state_data, 'value': value_data, 'move': move_data})
    permove_df.to_csv(PROCESSED_FNAME)
    permove_df = pd.read_csv(PROCESSED_FNAME)
    return permove_df


def generate_lichess_dataset():
    ELO_CUTOFF = LICHESS_CUTOFFS['top3%']
    pgn = open(PGN_PATH)
    stat, i = os.stat(PGN_PATH), 0
    high_elo_offsets = []
    offset = pgn.tell()
    headers = chess.pgn.read_headers(pgn)
    while headers is not None:
        white_elo, black_elo = int(
            headers['WhiteElo']), int(headers['BlackElo'])
        if min(white_elo, black_elo) > ELO_CUTOFF:
            high_elo_offsets.append(offset)

        i += 1
        offset = pgn.tell()
        headers = chess.pgn.read_headers(pgn)

        if i % 10000 == 0:
            print(f"{100.*offset/stat.st_size:.2f}%", end='\r')
    print(f"{100.0:.2f}% Found all games")

    def game2permove(game):
        result = game.headers['Result']
        value = 1 if result == '1-0' else -1 if result == '0-1' else 0
        board = game.board()
        data = []
        for move in game.mainline_moves():
            move_str = board.san(move)
            data.append((board.fen(), value, move_str))
            board.push(move)
            value *= -1
        return data

    state_data, value_data, move_data = [], [], []
    for i, offset in enumerate(high_elo_offsets):
        pgn.seek(offset)
        game = chess.pgn.read_game(pgn)
        for sd, vd, md in game2permove(game):
            state_data.append(sd)
            value_data.append(vd)
            move_data.append(md)
        if i % 100 == 0:
            print(f"{100.*i/len(high_elo_offsets):.2f}%", end='\r')
    print(f"{100.0:.2f}% completed processing games")

    permove_df = pd.DataFrame(
        data={'state': state_data, 'value': value_data, 'move': move_data})
    return permove_df


def __get_dataframe(path, generate):
    if not os.path.exists(path):
        print(f'{path} does not exist, generating dataframe')
        df = generate()
        df.to_csv(path)
    df = pd.read_csv(path)
    print(f'loaded dataframe {path}')
    return df


def get_dataframe():
    return __get_dataframe(PROCESSED_FNAME, generate_dataset)


def get_value_dist_dataframe():
    return __get_dataframe(VALUE_DIST_DF_FNAME, generate_value_dist_dataframe)


def get_lichess_dataframe():
    return __get_dataframe(LICHESS_DF_FNAME, generate_lichess_dataset)


def generate_value_dist_dataframe():
    state2value = collections.defaultdict(lambda: [0, 0, 0])
    df = get_dataframe()
    for row in df.itertuples(index=False):
        state, value, move = row.state, row.value, row.move
        state2value[state][value+1] += 1

    state_data, value_data, move_data = [], [], []
    for row in df.itertuples(index=False):
        state, _, move = row.state, row.value, row.move
        values = [v/w for v,
                  w in zip(state2value[state], [0.431, 0.114, 0.455])]
        l, t, w = [v/sum(values) for v in values]
        state_data.append(state)
        value_data.append(w - l)
        move_data.append(move)

    return pd.DataFrame(data={'state': state_data, 'value': value_data, 'move': move_data})


def get_dataloaders(batch_size, permove_df):
    n_rows = permove_df.shape[0]
    np.random.seed(42)
    idxs = np.arange(n_rows)
    np.random.shuffle(idxs)
    train, val, test = idxs[:int(
        n_rows*.9)], idxs[int(n_rows*.9):int(n_rows*.95)], idxs[int(n_rows*.95):]
    print((len(train), len(val), len(test)))
    train_ds, val_ds, test_ds = [SupervisedChessDataset(permove_df, idxs)
                                 for idxs in [train, val, test]]
    train_dl, val_dl, test_dl = [DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=get_num_workers_from_args())
                                 for ds in [train_ds, val_ds, test_ds]]
    return train_dl, val_dl, test_dl


def get_value_dist_dataloaders(batch_size, permove_df):
    n_rows = permove_df.shape[0]
    np.random.seed(42)
    idxs = np.arange(n_rows)
    np.random.shuffle(idxs)
    train, val, test = idxs[:int(
        n_rows*.9)], idxs[int(n_rows*.9):int(n_rows*.95)], idxs[int(n_rows*.95):]
    print((len(train), len(val), len(test)))
    train_ds, val_ds, test_ds = [SupervisedChessDataset(permove_df, idxs)
                                 for idxs in [train, val, test]]
    train_dl, val_dl, test_dl = [DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=get_num_workers_from_args())
                                 for ds in [train_ds, val_ds, test_ds]]
    return train_dl, val_dl, test_dl


class ChildDelegatedDataset(Dataset):
    def __init__(self, filenames, filename2len):
        self.filenames = filenames
        self.filename2len = filename2len
        self.length = sum((filename2len[f] for f in filenames))

    def __len__(self):
        return self.length

    def assign_child(self, worker_id, num_workers):
        self.child = ChildDataset(self, worker_id, num_workers)

    def __getitem__(self, idx):
        # NOTE this doesnt guarantee that all data will be sampled over an epoch.
        # Since we don't know how the Sampler will distribute the idx to each child
        return self.child[idx%len(self.child)]


class ChildDataset(Dataset):
    def __init__(self, parent_ds, worker_id, num_workers):
        data_per_child = math.ceil(len(parent_ds) / num_workers)
        first_idx, end_idx = data_per_child*worker_id, min(len(parent_ds), data_per_child*(worker_id+1))
        sub_data_frames, d_idx, f_idx = [], 0, 0
        while d_idx < end_idx:
            filename = parent_ds.filenames[f_idx]
            f_len = parent_ds.filename2len[filename]
            if d_idx+f_len > first_idx:
                c_idx_start = max(first_idx-d_idx, 0)
                c_idx_end = min(end_idx, d_idx + f_len) - d_idx
                df = pd.read_csv(filename).iloc[c_idx_start:c_idx_end]
                sub_data_frames.append(df)
            f_idx += 1
            d_idx += f_len
        self.df = pd.concat(sub_data_frames)
        self.length = self.df.shape[0]
        self.start = first_idx
        self.end = end_idx

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        row = self.df.iloc[idx, :]
        state, value, move = row.state, row.value, row.move
        board = Board(state)
        move_idxs = move2idx(board.parse_san(move))
        return {
            'state': board2tensor(board),
            'value': torch.Tensor([value]),
            'policy': state2policy(board, move_idxs),
        }


def worker_init_fn(worker_id):
    worker_info = torch.utils.data.get_worker_info()
    dataset = worker_info.dataset  # the dataset copy in this worker process
    dataset.assign_child(worker_id, worker_info.num_workers)
    print(f'worker {worker_id} granted data subset of length {len(dataset.child)}')
    # worker_info.dataset = ChildDataset(dataset, worker_id, worker_info.num_workers)
    # overall_start = 0
    # overall_end = dataset.length-1
    # # configure the dataset to only process the split workload
    # per_worker = int(
    #     math.ceil((overall_end - overall_start) / float(worker_info.num_workers)))
    # worker_id = worker_info.id
    # dataset.start = overall_start + worker_id * per_worker
    # dataset.end = min(dataset.start + per_worker, overall_end)


N_SPLIT = 1000
N_TRAIN = 900
N_VALID = 50
N_TEST = 50


def get_split_dataloaders(batch_size, get_df):
    fname2len_path = os.path.join(LICHESS_BROKEN_DF_FOLDER, 'fname2len.json')
    files = [os.path.join(LICHESS_BROKEN_DF_FOLDER, f'{i}.csv') for i in range(N_SPLIT)]

    if not os.path.exists(LICHESS_BROKEN_DF_FOLDER):
        os.mkdir(LICHESS_BROKEN_DF_FOLDER)

        df = get_df()
        n_rows = df.shape[0]
        rows_per_file = math.ceil(n_rows/N_SPLIT)
        fname2len = {}
        # shuffle dataframe in place
        df = df.sample(frac=1).reset_index(drop=True)
        print('writing dataframe to:', LICHESS_BROKEN_DF_FOLDER, 'this will take about 1 minute/GB')

        for i, path in enumerate(files):
            s, e = rows_per_file*i, min(rows_per_file*(i+1), n_rows)
            sub_df = df.iloc[s:e]
            sub_df.to_csv(path)
            fname2len[path] = e - s
        with open(fname2len_path, 'w') as f:
            json.dump(fname2len, f)
    else:
        with open(fname2len_path) as f:
            fname2len = json.load(f)
    
    print('dataframe on disk, delegating worker processes')
    train_ds, val_ds, test_ds = [
        ChildDelegatedDataset(files[s:e], fname2len) for s, e in 
        [
            (0, N_TRAIN),
            (N_TRAIN, N_TRAIN+N_VALID),
            (N_TRAIN+N_VALID, N_TRAIN+N_VALID+N_SPLIT),
        ]
    ]

    train_dl, val_dl, test_dl = [
        DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=True,
            num_workers=get_num_workers_from_args(),
            worker_init_fn=worker_init_fn,
        )
        for ds in [train_ds, val_ds, test_ds]
    ]
    return train_dl, val_dl, test_dl
