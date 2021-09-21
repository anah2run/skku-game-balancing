import gym
import numpy as np
import matplotlib.pyplot as plt

import sys
import random
from collections import deque
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.initializers import RandomUniform
from tensorflow.keras.optimizers import Adam


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
    env = gym.make('MountainCar-v0')
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n

    agent = DQNAgent(state_size, action_size)
    # 가중치를 복원합니다
    # agent.model.load_weights('./save_model/model')

    scores, episodes = [], []
    score_avg = 0

    num_episode = 300
    for e in range(num_episode):
        done = False
        score = 0
        max_position = -0.4

        state = env.reset()
        state = state.reshape(1, -1)
        while not done:
            env.render()

            action = agent.choose_action(state)

            next_state, _, done, info = env.step(action)

            next_state = next_state.reshape(1, -1)
            vel = next_state[0][1]
            pos = next_state[0][0]
            reward = ((pos + 0.5) * 10) ** 2 / 10
            if vel > 0:
                reward += 30 * vel

            finish = pos >= 0.5
            score += reward
            reward = 75 if finish else reward
            if pos > max_position:
                max_position = pos
            agent.remember(state, action, reward, next_state, done)

            if len(agent.memory) >= agent.train_start:
                agent.train_model()

            state = next_state

            if done:
                agent.update_target_model()

                score_avg = 0.9 * score_avg + 0.1 * score if score_avg != 0 else score
                print(
                    'episode: {:3d} | score avg {:3.2f} | memory length: {:4d} | epsilon: {:.4f} | max_pos: {:.2f}'.format(
                        e, score_avg, len(agent.memory), agent.epsilon, max_position))

                scores.append(score_avg)
                episodes.append(e)
                plt.plot(episodes, scores, 'b')
                plt.xlabel('episode')
                plt.ylabel('average score')
                plt.savefig('cartpole_graph.png')
                print(finish, done)
                if finish and score_avg > 300:
                    agent.save_model()
                    sys.exit()
