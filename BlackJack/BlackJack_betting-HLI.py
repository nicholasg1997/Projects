#!/usr/bin/env python
# coding: utf-8

import random

import gym
import numpy as np
from gym import spaces
from stable_baselines3 import PPO
from tqdm import tqdm


def count_(deck, *args):
    """takes in card numbers and counts sum of all cards"""
    counts = []
    for card in args:
        c = deck.count(card)
        counts.append(c)
    return sum(counts)


class BlackJack(gym.Env):
    """Black jack game for gym"""
    metadata = {'render.modes': ['console']}
    # 5 different betting multipliers
    STAY = 0
    HIT = 1

    def __init__(self, decks=1, wallet=2000):
        super(BlackJack, self).__init__()
        # model takes players hand sum, players first card, and low, medium, high cards percentage
        self.betting_model = PPO.load('data/PPO_BlackJack_Hit_Stay')
        # prep game deck based on how many decks in play (default=1)
        self.decks = decks
        self.deck = self.make_deck()
        # empty both dealer and players hand
        self.player_hand = []
        self.dealer_hand = []
        self.dealer_hand.append(self.draw())
        # setup action spaces
        # 2 actions, a small bet, and a big bet
        self.n_actions = 6
        self.action_space = spaces.Discrete(self.n_actions)
        # 12 observations are players hand, dealers face up card, every cards percentage
        self.observation_space = spaces.Box(low=-1000, high=1000,
                                            shape=(1,), dtype=np.float32)

    def reset(self):
        """reset the game by rebuilding the deck, 
        emptying both the player and dealers hand,
        setting dealer face up card to None,
        """
        self.player_hand = []
        self.dealer_hand = []
        card = self.draw()
        self.dealer_hand.append(card)

        hli_index = self.calculate_ratio()
        np.reshape(hli_index, (1,))

        return np.array(np.reshape(hli_index, (1,)), dtype=np.float32)

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
        # calculate high med low cards percentage
        low = count_(self.deck, 2, 3, 4, 5, 6)
        med = count_(self.deck, 7, 8, 9)
        high = count_(self.deck, 10, 1)
        r = (52 * self.decks) - (low + med + high)
        try:
            hli_index = 100 * ((low - high) / r)
        except ZeroDivisionError:
            hli_index = 0
        np.reshape(hli_index, (1,))
        return hli_index

    def step(self, action):
        self.reset()
        bet = action
        done = False

        low = round(count_(self.deck, 1, 2, 3) / len(self.deck), 2)
        med = round(count_(self.deck, 4, 5, 6, 7) / len(self.deck), 2)
        high = round(count_(self.deck, 8, 9, 10) / len(self.deck), 2)

        while not done:
            # get observation space
            if len(self.deck) < 15:
                self.deck = self.make_deck()
            hli_index = self.calculate_ratio()
            np.reshape(hli_index, (1,))
            # generate observation for hit-stay model
            obs = np.array([sum(self.player_hand), self.dealer_hand[0], low, med, high], dtype=np.float32)
            # get action from model
            action_, _ = self.betting_model.predict(obs, deterministic=True)

            if action_ == self.HIT:
                if len(self.deck) < 15:
                    self.deck = self.make_deck()
                card = self.draw()
                self.player_hand.append(card)
                if 1 in self.player_hand:
                    if sum(self.player_hand) + 10 <= 21:
                        self.player_hand.remove(1)
                        self.player_hand.append(11)
                if 11 in self.player_hand:
                    if sum(self.player_hand) > 21:
                        self.player_hand.remove(11)
                        self.player_hand.append(1)

            if action_ == self.STAY:
                # dealer plays until hand total is greater then or equal to 17
                while sum(self.dealer_hand) < 17:
                    if sum(self.dealer_hand) > sum(self.player_hand):
                        break
                    else:
                        card = self.draw()
                        self.dealer_hand.append(card)
                    if 1 in self.dealer_hand:
                        if sum(self.dealer_hand) + 10 <= 21:
                            self.dealer_hand.remove(1)
                            self.dealer_hand.append(11)
                    if 11 in self.dealer_hand:
                        if sum(self.dealer_hand) > 21:
                            self.dealer_hand.remove(11)
                            self.dealer_hand.append(1)
                    if len(self.deck) < 15:
                        self.deck = self.make_deck()

                done = True

        # decide who won the game       
        if sum(self.player_hand) <= 21:
            if sum(self.player_hand) == sum(self.dealer_hand):
                # tied
                win = 0
            elif sum(self.player_hand) > sum(self.dealer_hand) or sum(self.dealer_hand) > 21:
                win = 1
            else:
                # lost
                win = -1
        else:
            # went bust
            win = -1

        # calculate rewards
        # if player loses
        if win == -1:
            reward = bet * win
        # if player wins
        elif win == 1:
            reward = (bet * win) - (self.n_actions - 1)
        # player draws
        else:
            reward = 0

        info = {'player hand total:': self.player_hand,
                'dealer hand total:': self.dealer_hand,
                'length deck:': len(self.deck),
                'win:': win,
                'action:': bet}

        return np.array(np.reshape(hli_index, (1,)), dtype=np.float32), float(reward), done, info


from stable_baselines3.common.env_checker import check_env

env = BlackJack()
check_env(env)

model = PPO('MlpPolicy', env, verbose=1).learn(100_000)

profits = []
wins = []
actions = []
p = 0
obs = env.reset()
for i in tqdm(range(20_000)):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)
    p += action * int(info['win:'])
    wins.append(info['win:'])
    actions.append(action)
    profits.append(reward)

w = wins.count(1)
l = wins.count(-1)
w / (w + l)

print(sum(profits))
print(actions.count(0))
