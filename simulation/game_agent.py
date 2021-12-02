import math
import sys

import numpy as np
import random

import gym
from gym import spaces
from gym.utils import seeding


class AgentEnv(gym.Env):
    metadata = {"render.modes": ["human", "ansi"]}

    def __init__(self, name, params, cdt, enemy_max_hp):
        self.name = name
        self.skill_cdt = np.asarray(cdt)
        self.skill_num = len(cdt)
        self.enemy_max_hp = enemy_max_hp
        self.init_params = params
        self.max_hp, self.armor, self.str, self.crit, self.accuracy, self.avoid, self.atk_duration, self.reg_hp = self.init_params
        self.viewer = None
        self.seed()
        self.action_space = spaces.Discrete(1 + self.skill_num)  # wait, skill1, skill2
        high = [self.max_hp, self.enemy_max_hp]
        high.extend([1] * self.skill_num)
        self.low = np.array([0] * (2+self.skill_num), dtype=np.float32)  # my_hp, my_mp, enemy_hp, enemy_mp, skill1_timer, skill2_timer
        self.high = np.array(high, dtype=np.float32)
        self.observation_space = spaces.Box(self.low, self.high, dtype=np.float32)
        self.buff_q = []

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        assert self.action_space.contains(action), "%r (%s) invalid" % (action, type(action), )
        hp, enemy_hp, _, _ = self.state
        for i in range(self.skill_num):
            self.skill_timer[i] = max(0, self.skill_timer[i]-1)
        self.atk_timer = max(0, self.atk_timer-1)
        skill_ratio = self.skill_timer/self.skill_cdt
        self.state = [min(self.max_hp, hp + self.reg_hp * self.heal_ratio), enemy_hp, 0, 0]
        for i in range(self.skill_num):
            self.state[2+i] = skill_ratio[i]
        rmv_q = []
        buff_num = len(self.buff_q)
        for i in range(buff_num):
            buff_time = self.buff_q[i][0] = self.buff_q[i][0] - 1
            if  buff_time <= 0:
                rmv_q.append(i)
        while len(rmv_q) > 0:
            i = rmv_q.pop()
            self.remove_buff(i)
        reward = 0
        done = False
        return np.array(self.state, dtype=np.float32)  # , reward, done, {}

    def reset(self):
        self.max_hp, self.armor, self.str, self.crit, self.accuracy, self.avoid, self.atk_duration, self.reg_hp = self.init_params
        self.heal_ratio, self.deal_ratio = 1, 1
        self.skill_timer = np.zeros(self.skill_num)
        self.atk_timer = 0
        self.state = [self.max_hp, self.enemy_max_hp, 0, 0]
        self.buff_q = []
        return np.array(self.state, dtype=np.float32)

    def render(self, mode="human"):
        print("{0} state :".format(self.name), self.state, "/buff :", self.buff_q)
        return

    def get_keys_to_action(self):
        return {}

    def get_rand(self):
        return random.random()

    def use_skill(self, id):
        if self.skill_timer[id] <= 0:
            self.skill_timer[id] = self.skill_cdt[id]
            return True
        return False

    def take_damage(self, amount):
        if amount <= 0:
            return False
        if self.get_rand() < self.avoid:
            print("{0} avoid the damage".format(self.name))
            return False
        if amount <= self.armor:
            print("{0} blocked the damage".format(self.name))
            return False
        self.state[0] -= amount
        print("{0} has taken {1:5} damage".format(self.name, amount))
        return True

    def heal(self, amount):
        amount *= self.heal_ratio
        hp = self.state[0]
        hp_left = self.max_hp - hp
        self.state[0] = min(self.max_hp, hp + amount)
        print("{0} healed self {1:5}".format(self.name, min(hp_left, amount)))
        return

    def attack(self):
        if self.atk_timer <= 0:
            self.atk_timer = self.atk_duration
            return True
        return False

    def cal_amount(self, amount):
        if self.get_rand() < self.crit:
            amount *= 1.5
        return amount

    def give_damage(self, amount):
        amount *= self.deal_ratio
        if self.get_rand() > self.accuracy:
            # print("{0} missed attack".format(self.name))
            return 0

        # print("{0} dealt {1:5} damage to enemy".format(self.name, amount))
        return amount

    def set_enemy_hp(self, hp):
        self.state[1] = hp
        return True

    def add_buff(self,buff, time):
        self.buff_q.append([time, buff])
        for b in buff.keys():
            if b == 'deal_ratio':
                self.deal_ratio *= buff[b]
            elif b == 'heal_ratio':
                self.heal_ratio *= buff[b]
            elif b == 'armor':
                self.armor -= buff[b]
        return

    def remove_buff(self, index):
        buff = self.buff_q[index][1]
        for b in buff.keys():
            if b == 'deal_ratio':
                self.deal_ratio /= buff[b]
            elif b == 'heal_ratio':
                self.heal_ratio /= buff[b]
            elif b == 'armor':
                self.armor += buff[b]
        del self.buff_q[index]
        return


    def is_dead(self):
        return self.state[0] <= 0

    def hp_ratio(self):
        return self.state[0] / self.max_hp

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None
