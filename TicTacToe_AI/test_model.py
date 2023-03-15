import numpy as np
from stable_baselines3 import PPO
from TicTacToe_v2 import LearningTicTacToe
from TicTacToe import get_legal_moves

model = PPO.load("adversarial_tictactoe_model_3x3_PPO.zip")
ttt = LearningTicTacToe(model=model, train=False)

print("running...")
while not ttt.is_game_over()[0]:
    b = ttt.board.flatten()
    action, _states = model.predict(np.concatenate((b, np.array(get_legal_moves(b)))), deterministic=True)
    ttt.step(action)
    ttt.render()
    move = int(input("pick a position: "))
    print(ttt.legal_moves)
    if move not in ttt.legal_moves:
        print("illegal move")
        break
    ttt.step(move)
    if ttt.is_game_over()[0]:
        print(f"winner is {ttt.winner}")
