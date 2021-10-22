import gym
import random
from collections import deque
import tensorflow as tf
from tensorflow.keras.models import Sequential
import numpy as np
from tensorflow.keras.layers import Dense
from tensorflow.keras.initializers import RandomUniform
from tensorflow.keras.optimizers import Adam

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

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
        self.train_start = 100

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

    def save_model(self, name):
        self.target_model.compile(optimizer=self.optimizer,
                           loss='sparse_categorical_crossentropy',
                           metrics=['accuracy'])
        self.target_model.save('./save_model/'+name)

    def save_weight(self, name):
        self.target_model.save_weights('./save_model/'+name)

    def load_model(self, name):
        self.model = tf.keras.models.load_model('./save_model/'+name)
        self.target_model = tf.keras.models.load_model('./save_model/' + name)

    def load_weight(self, name):
        self.model.load_weights('./save_model/'+name)
        self.target_model.load_weights('./save_model/' + name)
