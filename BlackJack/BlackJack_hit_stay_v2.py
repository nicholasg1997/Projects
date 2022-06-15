#!/usr/bin/env python
# coding: utf-8


import numpy as np
import gym
from gym import spaces
import random
from tqdm import tqdm
from stable_baselines3 import DQN, PPO
from stable_baselines3.common.env_checker import check_env


def count_(deck, *args):
    """takes in card numbers and counts sum of all cards"""
    counts = []
    for card in args:
        c = deck.count(card)
        counts.append(c)
    return sum(counts)


class BlackJack(gym.Env):
    metadata = {'render.modes': ['console']}
    # two player actions stay:0 hit:1
    STAY = 0
    HIT = 1

    def __init__(self, decks=1, wallet=2000):
        super(BlackJack, self).__init__()
        # prep game deck based on howmany decks in play (default=1)
        self.decks = decks
        self.deck = self.make_deck()
        # empty both dealer and players hand
        self.player_hand = []
        self.dealer_hand = []
        self.dealer_hand.append(self.draw())
        # setup action spaces
        # there are two actions stay:0 or hit:1
        n_actions = 2
        self.action_space = spaces.Discrete(n_actions)
        # 5 observations are players hand, dealers face up card, low card%, med card%, high card%
        self.observation_space = spaces.Box(low=0, high=30,
                                            shape=(5,), dtype=np.float32)

    def reset(self):
        """reset the game by emptying both the player and dealers hand,
        setting dealer face up card to None, profit back to zero,
        """
        # self.deck = self.make_deck()
        self.player_hand = []
        self.dealer_hand = []
        card = self.draw()
        self.dealer_hand.append(card)

        low, med, high = self.calculate_ratio()
        return np.array([sum(self.player_hand), self.dealer_hand[0], low, med, high], dtype=np.float32)

    # make the deck of cards
    def make_deck(self):
        deck = [i for i in range(1, 10)] * 4
        face_cards = [10, 10, 10, 10] * 4
        deck = deck + face_cards
        deck = deck * self.decks
        return deck

    # draw a card from the deck then remove it from the deck
    def draw(self):
        card = random.choice(self.deck)
        self.deck.remove(card)
        return card

    def calculate_ratio(self):
        # calculte high med low cards percentage
        low = round(count_(self.deck, 1, 2, 3) / len(self.deck), 2)
        med = round(count_(self.deck, 4, 5, 6, 7) / len(self.deck), 2)
        high = round(count_(self.deck, 8, 9, 10) / len(self.deck), 2)
        return low, med, high

    def step(self, action):
        if len(self.deck) < 15:
            self.deck = self.make_deck()
        low, med, high = self.calculate_ratio()
        draw = False

        if action == self.HIT:
            if len(self.deck) < 15:
                self.deck = self.make_deck()
            card = self.draw()
            self.player_hand.append(card)
            done = False
            if sum(self.player_hand) > 21:
                done = True

        if action == self.STAY:
            # dealer plays until hand total is greater then or equal to 17
            while sum(self.dealer_hand) < 17:
                if sum(self.dealer_hand) > sum(self.player_hand):
                    break
                else:
                    card = self.draw()
                    self.dealer_hand.append(card)
                if len(self.deck) < 15:
                    self.deck = self.make_deck()

            done = True

        if done:
            # check that player is lower then 22
            if sum(self.player_hand) <= 21:
                # higher total then dealer
                if sum(self.player_hand) > sum(self.dealer_hand) or sum(self.dealer_hand) > 21:
                    # player won
                    win = True

                elif sum(self.player_hand) == sum(self.dealer_hand):
                    # draw
                    win = False
                    draw = True
                else:
                    win = False
            else:
                # player went bust
                win = False
            reward = 1.0 if win else -1.0  # win:1 loss:-1
            if draw:
                reward = 0.5
            info = {'player_hand': sum(self.player_hand),
                    'dealer_hand': sum(self.dealer_hand),
                    'len deck:': len(self.deck)
                    }
            self.reset()
            return np.array([sum(self.player_hand), self.dealer_hand[0], low, med, high],
                            dtype=np.float32), reward, done, info

        reward = 0.0
        info = {'player_hand': sum(self.player_hand),
                'dealer_hand': sum(self.dealer_hand),
                'len deck:': len(self.deck)}
        return np.array([sum(self.player_hand), self.dealer_hand[0], low, med, high],
                        dtype=np.float32), reward, done, info

env = BlackJack()
check_env(env)

# Instantiate the env
env = BlackJack()

model = PPO('MlpPolicy', env, verbose=0).learn(500_000)

wins = []
n_steps = 20
for i in tqdm(range(10_000)):
    done = False
    obs = env.reset()
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        wins.append(reward)

l = wins.count(-1)
w = wins.count(1)
d = wins.count(0.5)
print(w / (l + w + d))
