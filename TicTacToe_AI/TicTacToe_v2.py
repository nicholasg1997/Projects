import gym
from gym import spaces
from stable_baselines3 import PPO
import numpy as np
import random
from stable_baselines3.common.vec_env import DummyVecEnv
from TicTacToe import get_legal_moves

# create the environment
class LearningTicTacToe(gym.Env):
    """Tic Tac Toe game environment"""
    # training against PPO Model

    BOARD_SIZE = 3
    # reward values
    TIE_REWARD = 0.5
    WIN_REWARD = 1
    LOSE_REWARD = -1
    ILLEGAL = -2
    MOVE = 0.1  # reward for placing on a empty spot

    def __init__(self, model, train: bool = True):
        self.model = model

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

            # have player two move based on model provided
            if self.train:
                action, _states = self.model.predict(
                    np.concatenate((self.board.flatten(), np.array(get_legal_moves(self.board)))), deterministic=True)

                row = action // self.BOARD_SIZE
                col = action % self.BOARD_SIZE
                if self.board[row][col] != 0:
                    return np.concatenate((self.board.flatten(), np.array(get_legal_moves(self.board)))), 0, True, {}
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
            if self.train:
                self.reset()  # call reset function to reset the environment
            self.reset()  # call reset function to reset the environment
            return obs, reward, done, {}

    def reset(self):
        self.board = np.zeros((self.BOARD_SIZE, self.BOARD_SIZE))
        self.player = 1
        self.legal_moves = list(range(self.BOARD_SIZE ** 2))  # Reset legal moves
        return np.concatenate((self.board.flatten(), np.array(get_legal_moves(self.board))))

    def render(self, mode='human'):
        print(self.board)

    def is_game_over(self):
        # check if rows are complete
        # returns if game is over and if it ended in a tie
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
    # load the model
    trained_model = PPO.load("tictactoe_model_3x3_PPO")
    for _ in range(5):
        env = LearningTicTacToe(trained_model)
        model = PPO('MlpPolicy', env, verbose=1)
        model.learn(total_timesteps=10_000)
        trained_model = model

    # save the model
    trained_model.save("adversarial_tictactoe_model_3x3_PPO")

