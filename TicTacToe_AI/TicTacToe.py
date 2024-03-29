import gym
from gym import spaces
from stable_baselines3 import PPO
import numpy as np
import random
from stable_baselines3.common.vec_env import DummyVecEnv


def get_legal_moves(board):
    legal_moves = [1 if x == 0 else 0 for x in board.flatten()]
    return legal_moves


class TicTacToe(gym.Env):
    """Tic Tac Toe game environment"""

    BOARD_SIZE = 3
    TIE_REWARD = 0.5
    WIN_REWARD = 1
    LOSE_REWARD = -1
    ILLEGAL = -2
    MOVE = 0.1  # reward for placing on a empty spot

    def __init__(self, train: bool = True):
        self.action_space = spaces.Discrete(self.BOARD_SIZE ** 2)
        self.observation_space = spaces.Box(low=-1, high=1,
                                            shape=(self.BOARD_SIZE ** 2 + self.BOARD_SIZE ** 2,),
                                            dtype=np.float32)

        self.board = np.zeros((self.BOARD_SIZE, self.BOARD_SIZE))
        self.player = 1
        self.legal_moves = list(range(self.BOARD_SIZE ** 2))
        self.winner = None
        self.train = train

    def step(self, action: int):
        row = action // self.BOARD_SIZE
        col = action % self.BOARD_SIZE
        if self.board[row][col] == 0:
            self.board[row][col] = self.player
            done, tie = self.is_game_over()
            reward = self.get_reward(done, tie)
            self.player = -self.player

            if done:
                return np.concatenate((self.board.flatten(), np.array(get_legal_moves(self.board)))), reward, done, {}

            if not self.legal_moves:
                done = True
                reward = self.TIE_REWARD
                return np.concatenate((self.board.flatten(), np.array(get_legal_moves(self.board)))), reward, done, {}

            # Update legal moves
            self.legal_moves.remove(action)

            # have player two move randomly
            if self.train:
                action = random.choice(self.legal_moves)
                row = action // self.BOARD_SIZE
                col = action % self.BOARD_SIZE
                self.board[row][col] = self.player
                done, tie = self.is_game_over()
                reward = self.get_reward(done, tie)
                self.player = -self.player

                if done:
                    return np.concatenate(
                        (self.board.flatten(), np.array(get_legal_moves(self.board)))), reward, done, {}

            # out of moves(tie)
            if not self.legal_moves:
                done = True
                reward = self.TIE_REWARD

                return np.concatenate((self.board.flatten(), np.array(get_legal_moves(self.board)))), reward, done, {}

            return np.concatenate((self.board.flatten(), np.array(get_legal_moves(self.board)))), reward, done, {}

        # Invalid move, penalize and end the game
        else:
            done = True
            reward = self.ILLEGAL
            obs = np.concatenate((self.board.flatten(), np.array(get_legal_moves(self.board))))
            self.reset()  # call reset function to reset the environment
            return obs, reward, done, {}

    def reset(self):
        self.board = np.zeros((self.BOARD_SIZE, self.BOARD_SIZE))
        self.player = 1
        self.legal_moves = list(range(self.BOARD_SIZE ** 2))
        return np.concatenate((self.board.flatten(), np.array(get_legal_moves(self.board))))

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

    def get_reward(self, done: bool, tie: bool) -> int:
        if done:
            if tie:
                return self.TIE_REWARD
            elif self.winner == 1:
                return self.WIN_REWARD
            else:
                return self.LOSE_REWARD
        else:
            return self.MOVE


if __name__ == '__main__':
    env = TicTacToe()
    env = DummyVecEnv([lambda: env])

    model = PPO('MlpPolicy', env, verbose=1)
    model.learn(total_timesteps=50_000)
    model.save("tictactoe_model_3x3_PPO")

    play = False
    if play:
        ttt = TicTacToe(train=False)
        while not ttt.is_game_over()[0]:
            b = ttt.board.flatten()
            action, _states = model.predict(b, deterministic=True)
            ttt.step(action)
            ttt.render()
            move = int(input("pick a position: "))
            ttt.step(move)

