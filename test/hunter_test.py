import gym
import numpy as np
# import matplotlib.pyplot as plt
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import sys
import random
from collections import deque
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.initializers import RandomUniform
from tensorflow.keras.optimizers import Adam

from hunter import HunterEnv


# 24?
class DQN(tf.keras.Model):
    def __init__(self, action_size):
        super(DQN, self).__init__()
        self.fc1 = Dense(24, activation='relu')
        self.fc2 = Dense(24, activation='relu')
        self.fc_out = Dense(action_size, kernel_initializer=RandomUniform(-1e-3, 1e-3))

    def call(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        q = self.fc_out(x)
        return q


class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.discount_factor = 0.99
        self.learning_rate = 0.001
        self.epsilon = 1.0
        self.epsilon_decay = 0.999
        self.epsilon_min = 0.01
        self.batch_size = 64
        self.train_start = 1000

        self.memory = deque(maxlen=2000)

        self.model = DQN(action_size)
        self.target_model = DQN(action_size)
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)

        self.update_target_model()
        self.model.compile(optimizer=self.optimizer,
                           loss='sparse_categorical_crossentropy',
                           metrics=['accuracy'])

    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def choose_action(self, state):
        return random.randrange(self.action_size) if (np.random.rand() <= self.epsilon) else np.argmax(
            self.model.predict(state))

    def train_model(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        mini_batch = random.sample(self.memory, self.batch_size)
        states = np.array([sample[0][0] for sample in mini_batch])
        # array([[-0.00457641, -0.00932114,  0.04361066,  0.15769866]])
        actions = np.array([sample[1] for sample in mini_batch])
        # 1
        rewards = np.array([sample[2] for sample in mini_batch])
        # 0.1
        next_states = np.array([sample[3][0] for sample in mini_batch])
        # array([[-0.00457641, -0.00932114,  0.04361066,  0.15769866]])
        dones = np.array([sample[4] for sample in mini_batch])
        # false

        model_params = self.model.trainable_variables
        with tf.GradientTape() as tape:
            predicts = self.model(states)
            one_hot_action = tf.one_hot(actions, self.action_size)
            # [0,0,1,0] [0,1,0,0] [1,0,0,0]
            predicts = tf.reduce_sum(one_hot_action * predicts, axis=1)

            target_predicts = self.target_model(next_states)
            target_predicts = tf.stop_gradient(target_predicts)

            max_q = np.amax(target_predicts, axis=-1)
            targets = rewards + (1 - dones) * self.discount_factor * max_q
            loss = tf.reduce_mean(tf.square(targets - predicts))

        grads = tape.gradient(loss, model_params)
        self.optimizer.apply_gradients(zip(grads, model_params))

    def save_model(self):
        self.model.compile(optimizer=self.optimizer,
                           loss='sparse_categorical_crossentropy',
                           metrics=['accuracy'])
        self.model.save('./save_model/model_save')


if __name__ == "__main__":

    skill_set = {
        0: ('range_attack', 1),
        1: ('shot', 7),
        2: ('heal', 10)
    }
    max_hp = 100
    enemy_max_hp = 500
    init_pos = 1
    attack_range = 0.3
    mov_spd = 0.1
    enemy_pos = -1
    enemy_mov_spd = 0.03
    hunter_params = (max_hp, mov_spd, init_pos, (0, 1, 2))
    env = HunterEnv(hunter_params, skill_set)
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    print('s_size', state_size, 'a_size', action_size)
    agent = DQNAgent(state_size, action_size)
    # 가중치를 복원합니다
    # agent.model.load_weights('./save_model/model')

    scores, episodes = [], []
    score_avg = 0

    num_episode = 300
    for e in range(num_episode):
        done = False

        pos = init_pos
        enemy_hp = enemy_max_hp
        enemy_pos = -1
        state = env.reset(enemy_pos)
        score = 0
        while not done:
            state = state.reshape(1, -1)
            env.render()
            action = agent.choose_action(state)

            hp = env.hp
            pos = env.pos
            # print(action)
            dist = abs(pos-enemy_pos)
            reward = dist*2
            if action == 0:
                print('hunter stop')
            elif action == 1:
                print('hunter move left')
                reward = -1
                pos = max(-1, pos-mov_spd)
            elif action == 2:
                print('hunter move right')
                reward = -1
                pos = min(1, pos+mov_spd)
            elif action == 3:
                if env.use_skill(0) and dist <= attack_range:
                    enemy_hp -= 20
                    reward = 2
                    print('hunter attack')
            elif action == 4:
                if env.use_skill(1) and dist <= attack_range:
                    reward = 5
                    enemy_hp -= 50
                    print('hunter skill')
            elif action == 5:
                if env.use_skill(2):
                    heal = min(max_hp-hp, 20)
                    reward = heal /10
                    hp += heal
                    print('hunter heal')
            next_state, _, done, info = env.step(action)
            if enemy_pos > pos:
                enemy_pos -= enemy_mov_spd
            else:
                enemy_pos += enemy_mov_spd
            if abs(enemy_pos - pos) < 0.3:
                hp -= 1
                reward -= 1


            next_state = np.array((
                hp/max_hp,
                pos,
                enemy_hp/enemy_max_hp,
                enemy_pos,
                env.skill_timer[0] == 0,
                env.skill_timer[1] == 0,
                env.skill_timer[2] == 0
            ), dtype=np.float32)
            print(next_state)
            if enemy_hp < 0:
                reward = 100
            score += reward
            agent.remember(state, action, reward, next_state.reshape(1, -1), done)
            if len(agent.memory) >= agent.train_start:
                agent.train_model()
            state = next_state
            env.state = state
            env.hp = hp
            env.pos = pos
            if done:
                print('done!', score)

                agent.update_target_model()
                print(agent.model.predict(state))
                episodes.append(e)
                # plt.plot(episodes, scores, 'b')
                # plt.xlabel('episode')
                # plt.ylabel('average score')
                # plt.savefig('cartpole_graph.png')
                agent.save_model()
                sys.exit()
