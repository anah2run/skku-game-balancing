import numpy as np
import matplotlib.pyplot as plt
import os
import sys
from ml_agent import DQNAgent
from hunter import HunterEnv

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# 24?

if __name__ == "__main__":

    skill_set = {
        0: ('range_attack', 1, .3, 20),
        1: ('shot', 7, .4, 50),
        2: ('heal', 10, -1, .2),
        3: ('melee_attack', 1, .1, 30),
        4: ('charge', 10, .4, 10),
        5: ('heal2', 7, -1, .1),
    }
    hunter_max_hp = 100
    hunter_skill_set = (0, 1, 2)
    hunter_range = .3
    hunter_init_pos = 1
    hunter_mov_spd = 0.1
    warrior_pos = -1
    warrior_mov_spd = 0.9
    warrior_max_hp = 500
    hunter_params = (hunter_max_hp, hunter_mov_spd, hunter_init_pos, hunter_skill_set)
    # warrior_params = (max_hp, mov_spd, init_pos, (0, 1, 2))
    env = HunterEnv(hunter_params, skill_set)
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    print('s_size', state_size, 'a_size', action_size)
    agent = DQNAgent(state_size, action_size)
    # agent.target_model = agent.load_model('hunter')


    scores, episodes = [], []
    score_avg = 0

    num_episode = 3000
    for e in range(num_episode):
        done = False

        enemy_hp = warrior_max_hp
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
            reward = min(dist, hunter_range)*5
            if action == 0:
                print('hunter stop')
            elif action == 1:
                # print('hunter move left')
                pos = max(-1, pos-hunter_mov_spd)
            elif action == 2:
                # print('hunter move right')
                pos = min(1, pos+hunter_mov_spd)
            elif action == 3:
                if env.use_skill(0, dist):
                    enemy_hp -= 20
                    reward += 1
                   # print('hunter attack')
                else:
                    reward -= 1
            elif action == 4:
                if env.use_skill(1, dist):
                    reward = 2
                    enemy_hp -= 50
                   # print('hunter skill')
                else:
                    reward -= 1
            elif action == 5:
                if env.use_skill(2):
                    heal = min(hunter_max_hp-hp, 20)
                    reward = heal /10
                    hp += heal
                   # print('hunter heal')
            next_state, _, done, info = env.step(action)
            if enemy_pos > pos:
                enemy_pos = max(enemy_pos - warrior_mov_spd, pos)
            else:
                enemy_pos = min(enemy_pos + warrior_mov_spd, pos)
            if abs(enemy_pos - pos) < 0.3:
                hp -= 2
                reward -= 1


            next_state = np.array((
                hp/hunter_max_hp,
                pos,
                enemy_hp/warrior_max_hp,
                enemy_pos,
                env.skill_timer[0] == 0,
                env.skill_timer[1] == 0,
                env.skill_timer[2] == 0
            ), dtype=np.float32)
            # print(next_state)
            if enemy_hp < 0:
                reward = 100
            elif hp < 0:
                reward = -100
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
                print(state)

                episodes.append(e)
                # plt.plot(episodes, scores, 'b')
                # plt.xlabel('episode')
                # plt.ylabel('average score')
                # plt.savefig('cartpole_graph.png')
                if score > 1000 and enemy_hp < 0:
                    agent.save_model('hunter')
                    sys.exit()
