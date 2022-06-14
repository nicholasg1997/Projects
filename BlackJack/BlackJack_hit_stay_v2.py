#!/usr/bin/env python
# coding: utf-8


import numpy as np
import gym
from gym import spaces
import random
from tqdm import tqdm


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
        one = round(self.deck.count(1) / len(self.deck), 2)
        two = round(self.deck.count(2) / len(self.deck), 2)
        three = round(self.deck.count(3) / len(self.deck), 2)
        four = round(self.deck.count(4) / len(self.deck), 2)
        five = round(self.deck.count(5) / len(self.deck), 2)
        six = round(self.deck.count(6) / len(self.deck), 2)
        seven = round(self.deck.count(7) / len(self.deck), 2)
        eight = round(self.deck.count(8) / len(self.deck), 2)
        nine = round(self.deck.count(9) / len(self.deck), 2)
        ten = round(self.deck.count(10) / len(self.deck), 2)
        low = sum([one, two, three])
        med = sum([four, five, six, seven])
        high = sum([eight, nine, ten])
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


# In[43]:


from stable_baselines3.common.env_checker import check_env

env = BlackJack()
# It will check your custom environment and output additional warnings if needed
check_env(env)

# In[ ]:


# In[44]:


from stable_baselines3 import DQN, PPO
from stable_baselines3.common.cmd_util import make_vec_env

# Instantiate the env
env = BlackJack()
# env = make_vec_env(lambda: env, n_envs=3)


# In[45]:


model = PPO('MlpPolicy', env, verbose=0).learn(500_000)

# In[48]:


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
w / (l + w + d)


