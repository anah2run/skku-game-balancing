# q_learning.py

import gym
import numpy as np
import matplotlib.pyplot as plt
from gym.envs.registration import register
import random as pr

register(
    id='FrozenLake-v3',
    entry_point='gym.envs.toy_text:FrozenLakeEnv',
    kwargs={'map_name': '4x4',
            'is_slippery': False}
)

env = gym.make('FrozenLake-v3')

# Initialize table with all zeros
Q = np.zeros([env.observation_space.n, env.action_space.n])

'''1. Q 값이 업데이트될 때 maxQ(s',a') 에 곱할 감마 값을 설정한다.'''
dis = .99

num_episodes = 2000

# create lists to contain total rewards and steps per episode
rList = []
for i in range(num_episodes):
    # Reset environment and get first new observation
    state = env.reset()
    rAll = 0
    done = False
    '''2. E-Greedy 를 위한 확률값을 만들어준다. (step i이 지남에 따라 decay 되도록 설정)'''
    e = 1. / ((i // 100) + 1)

    # The Q-Table learning algorithm : 한번 수행할 때 마다 Q 한칸 업데이트
    while not done:

        '''E-Greedy 를 따라 작은 확률로 랜덤하게 가고, 큰 확률로 높은 Q 를 따르는 쪽으로 간다.'''
        if np.random.rand(1) < e:
            action = env.action_space.sample()
        else:
            action = np.argmax(Q[state, :])

        # Get new state and reward from environment
        new_state, reward, done, _ = env.step(action)

        # Update Q-Table with new knowledge using learning rate
        Q[state, action] = reward + dis * np.max(Q[new_state, :])

        rAll += reward
        state = new_state

    rList.append(rAll)

print("Success rate: " + str(sum(rList) / num_episodes))
print("Final Q-Table Values")
print(Q)
plt.bar(range(len(rList)), rList, color="blue")
plt.show()