import gym
from gym import spaces
from stable_baselines3 import PPO
import numpy as np
import random
from stable_baselines3.common.vec_env import DummyVecEnv


class TicTacToe(gym.Env):
    """Tic Tac Toe game environment
    changing train to False will allow you to play against the AI"""

    BOARD_SIZE = 3
    TIE_REWARD = 10
    WIN_REWARD = 50
    LOSE_REWARD = -10

    def __init__(self, train: bool = True):
        self.action_space = spaces.Discrete(self.BOARD_SIZE ** 2)
        self.observation_space = spaces.Box(low=-1, high=1, shape=(self.BOARD_SIZE ** 2,), dtype=np.float32)
        self.board = np.zeros((self.BOARD_SIZE, self.BOARD_SIZE))
        self.player = 1
        self.legal_moves = list(range(self.BOARD_SIZE ** 2))
        self.winner = None
        self.train = train

    def step(self, action: int):
        row = action // self.BOARD_SIZE
        col = action % self.BOARD_SIZE
        if action in self.legal_moves:
            self.board[row][col] = self.player
            done, tie = self.is_game_over()
            reward = self.get_reward(done, tie)
            self.player = -self.player

            if done:
                return self.board.flatten(), reward, done, {}

            if not self.legal_moves:
                done = True
                reward = self.TIE_REWARD
                return self.board.flatten(), reward, done, {}

            # Update legal moves
            self.legal_moves.remove(action)

            # have player two move randomly
            if self.train:
                action = random.choice(self.legal_moves)
                row = action // self.BOARD_SIZE
                col = action % self.BOARD_SIZE
                self.board[row][col] = self.player
                self.player = -self.player

            # out of moves(tie)
            if not self.legal_moves:
                done = True
                reward = self.TIE_REWARD

                return self.board.flatten(), reward, done, {}

            return self.board.flatten(), reward, done, {}

            # Invalid move, penalize and end the game
        else:
            return self.board.flatten(), self.LOSE_REWARD, True, {}

    def reset(self):
        self.board = np.zeros((self.BOARD_SIZE, self.BOARD_SIZE))
        self.player = 1
        self.legal_moves = list(range(self.BOARD_SIZE ** 2))  # Reset legal moves
        return self.board.flatten()

    def render(self, mode='human'):
        print(self.board)

    def is_game_over(self):
        # check if rows are complete
        for i in range(self.BOARD_SIZE):
            if abs(sum(self.board[i])) == self.BOARD_SIZE:
                self.winner = int(np.sign(sum(self.board[i])))
                return True, 0

        # check cols
        for i in range(self.BOARD_SIZE):
            if abs(sum(self.board[:, i])) == self.BOARD_SIZE:
                self.winner = int(np.sign(sum(self.board[:, i])))
                return True, 0

                # check negative diagonal
        if abs(sum(np.diagonal(self.board))) == self.BOARD_SIZE:
            self.winner = int(np.sign(sum(np.diagonal(self.board))))
            return True, 0

        # check negative diagonal
        if abs(sum(np.diagonal(np.fliplr(self.board)))) == self.BOARD_SIZE:
            self.winner = int(np.sign(sum(np.diagonal(np.fliplr(self.board)))))
            return True, 0

        # check if draw
        if not self.legal_moves:
            self.winner = 0
            return True, 1

        return False, 0

    def get_reward(self, done: bool, tie: int) -> int:
        if done:
            if tie:
                return self.TIE_REWARD
            elif self.winner == 1:
                return self.WIN_REWARD
            else:
                return self.LOSE_REWARD
        else:
            return 0


env = TicTacToe
env = DummyVecEnv([lambda: env])

model = PPO('MlpPolicy', env, verbose=1)
model.learn(total_timesteps=100_000)

ttt = TicTacToe(train=False)
while not ttt.is_game_over()[0]:
    b = ttt.board.flatten()
    action, _states = model.predict(b, deterministic=True)
    ttt.step(action)
    ttt.render()
    move = int(input("pick a position: "))
    ttt.step(move)
