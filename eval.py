# Evaluation model

from random_board_generator import generate_random_board
import chess.engine
import pandas as pd

positions = [generate_random_board(max_depth=50) for _ in range(200)]
evals = []

# Loading stockfish

engine = chess.engine.SimpleEngine.popen_uci("./stockfish/15/stockfish-windows-2022-x86-64-avx2.exe")

for position in positions:
    evals.append(engine.analyse(position, chess.engine.Limit(depth=18))['score'].white().score())
    print(engine.analyse(position, chess.engine.Limit(depth=18))['score'].white().score())


# Creating the dataframe to train the model

df = pd.DataFrame(columns=['position', 'eval'])

df['position'] = positions
df['eval'] = evals

print(df.head())