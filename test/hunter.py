import math
import sys

import numpy as np

import gym
from gym import spaces
from gym.utils import seeding


class HunterEnv(gym.Env):
    """
    Description:
        The agent (a car) is started at the bottom of a valley. For any given
        state the agent may choose to accelerate to the left, right or cease
        any acceleration.
    Source:
        The environment appeared first in Andrew Moore's PhD Thesis (1990).
    Observation:
        Type: Box(2)
        Num    Observation               Min            Max
        0      Car Position              -1.2           0.6
        1      Car Velocity              -0.07          0.07
    Actions:
        Type: Discrete(3)
        Num    Action
        0      Accelerate to the Left
        1      Don't accelerate
        2      Accelerate to the Right
        Note: This does not affect the amount of velocity affected by the
        gravitational pull acting on the car.
    Reward:
         Reward of 0 is awarded if the agent reached the flag (position = 0.5)
         on top of the mountain.
         Reward of -1 is awarded if the position of the agent is less than 0.5.
    Starting State:
         The position of the car is assigned a uniform random value in
         [-0.6 , -0.4].
         The starting velocity of the car is always assigned to 0.
    Episode Termination:
         The car position is more than 0.5
         Episode length is greater than 200
    """

    metadata = {"render.modes": ["human", "ansi"]}

    def __init__(self, hunter_params, skill_dict):
        self.max_hp, self.mov_spd, self.init_pos, self.skill_set = hunter_params
        self.skill_cool = [skill_dict[i][1] for i in self.skill_set]
        skill_num = len(self.skill_set)
        self.skill_timer = [0 for _ in range(skill_num)]
        self.viewer = None
        self.action_space = spaces.Discrete(6)
        low_base = [0, -1, 0, -1]
        low_base.extend([0] * skill_num)
        self.low = np.array(low_base, dtype=np.float32)
        self.high = np.array([1] * (4 + skill_num), dtype=np.float32)
        self.observation_space = spaces.Box(self.low, self.high, dtype=np.float32)
        self.hp = self.max_hp
        self.pos = self.init_pos
        self.seed()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        assert self.action_space.contains(action), "%r (%s) invalid" % (
            action,
            type(action),
        )
        self.skill_timer = np.add(self.skill_timer, -0.5)
        self.skill_timer = np.clip(self.skill_timer, 0, self.skill_cool)
        hp_ratio, self.pos, enemy_hp_ratio, self.enemy_pos, _, _, _ = self.state
        reward = -1.0

        done = bool(hp_ratio < 0 or enemy_hp_ratio < 0)

        # self.state = (hp_ratio, pos, enemy_hp_ratio, enemy_pos, 0, 0, 0)
        return np.array(self.state, dtype=np.float32), reward, done, {}

    def reset(self, enemy_pos):
        # self.np_random.uniform(low=-0.6, high=0.6)
        self.hp = self.max_hp
        self.pos = self.init_pos
        self.state = np.array([1, self.init_pos, 1, enemy_pos, 1, 1, 1])
        return np.array(self.state, dtype=np.float32)

    def render(self, mode="human"):
        return

    def get_keys_to_action(self):
        # Control with left and right arrow keys. example {(): 1, (276,): 0, (275,): 2, (275, 276): 1}
        return {}

    def use_skill(self, id, dist=0):
        if self.skill_timer[id] <= 0 and (self.skill_set[2] <= 0 or self.skill_set[2] >= dist):
            self.skill_timer[id] = self.skill_cool[id]
            return True
        return False

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None
