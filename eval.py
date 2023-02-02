# Evaluation model

from random_board_generator import generate_random_board
import chess.engine
import pandas as pd
import numpy as np

positions = [generate_random_board(max_depth=50) for _ in range(200)]
evals = []

# Loading stockfish

engine = chess.engine.SimpleEngine.popen_uci(
    "./stockfish/15/stockfish-windows-2022-x86-64-avx2.exe")

for position in positions:
    evals.append(engine.analyse(position, chess.engine.Limit(depth=18))[
                 'score'].white().score())
    print(engine.analyse(position, chess.engine.Limit(depth=18))
          ['score'].white().score())


engine.close()

# Creating the dataframe to train the model

df = pd.DataFrame(columns=['position', 'eval'])

df['position'] = positions
df['eval'] = evals

print(df.head())

# Now we need to convert the board representation to something meaningful.
# A 3d matrix of sizes (8 x 8 x 14) where 8x8 repersents the board and the 14 represents the 6 different pieces

squares_index = {
    'a': 0,
    'b': 1,
    'c': 2,
    'd': 3,
    'e': 4,
    'f': 5,
    'g': 6,
    'h': 7
}

# example: h3 -> 17


def square_to_index(square):
    letter = chess.square_name(square)
    return 8 - int(letter[1]), squares_index[letter[0]]


def split_dims(board):
    # this is the 3d matrix

    board3d = np.zeros((14, 8, 8), dtype=np.int8)

    # here we add the pieces's view on the matrix

    for piece in chess.PIECE_TYPES:
        for square in board.pieces(piece, chess.WHITE):
            idx = np.unravel_index(square, (8, 8))
            board3d[piece - 1][7 - idx[0]][idx[1]] = 1
        for square in board.pieces(piece, chess.BLACK):
            idx = np.unravel_index(square, (8, 8))
            board3d[piece + 5][7 - idx[0]][idx[1]] = 1

    # add attacks and valid moves too
    # so the network knows what is being attacked
    aux = board.turn
    board.turn = chess.WHITE
    for move in board.legal_moves:
        i, j = square_to_index(move.to_square)
        board3d[12][i][j] = 1
    board.turn = chess.BLACK
    for move in board.legal_moves:
        i, j = square_to_index(move.to_square)
        board3d[13][i][j] = 1
    board.turn = aux

    return board3d

# Modifying the dataset (adding a board_split_dims column)


df['board_split_dims'] = [split_dims(board) for board in df['position']]

print(df.head())

# Now we can drop the position column

df = df.drop('position')
