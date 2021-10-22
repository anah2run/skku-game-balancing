import numpy as np
import matplotlib.pyplot as plt
import os
import sys
from ml_agent import DQNAgent
from game_agent import HunterEnv

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

if __name__ == "__main__":

    skill_set = {
        0: ('range_attack', 1, .3, 20),
        1: ('shot', 7, .4, 50),
        2: ('heal', 10, -1, .2),
        3: ('melee_attack', 2, .1, 10),
        4: ('charge', 15, .4, 10),
        5: ('heal2', 7, -1, .1),
    }

    hunter_skill_set = (0, 1, 2)
    hunter_max_hp = 300
    hunter_range = .3
    hunter_init_pos = 1
    hunter_mov_spd = 0.16

    warrior_skill_set = (3, 4, 5)
    warrior_max_hp = 300
    warrior_range = .1
    warrior_init_pos = -1
    warrior_mov_spd = 0.09

    hunter_params = (hunter_max_hp, hunter_mov_spd, hunter_init_pos, hunter_skill_set)
    warrior_params = (warrior_max_hp, warrior_mov_spd, warrior_init_pos, warrior_skill_set)
    hunter_env = HunterEnv(hunter_params, skill_set)
    warrior_env = HunterEnv(warrior_params, skill_set)

    state_size = hunter_env.observation_space.shape[0]
    action_size = hunter_env.action_space.n

    print('s_size', state_size, 'a_size', action_size)
    hunter_agent = DQNAgent(state_size, action_size)
    hunter_agent.load_model('hunter')
    warrior_agent = DQNAgent(state_size, action_size)
    warrior_agent.load_model('warrior')
    scores, episodes = [], []
    score_avg = 0

    num_episode = 1000
    for e in range(num_episode):
        done = False

        warrior_hp = warrior_env.max_hp
        warrior_pos = warrior_env.init_pos
        hunter_hp = hunter_env.hp
        hunter_pos = hunter_env.init_pos

        hunter_state = hunter_env.reset(warrior_pos)
        warrior_state = warrior_env.reset(hunter_pos)
        hunter_score = 0
        warrior_score = 0
        score = 0
        cnt = 0
        while not done:
            cnt += 1
            # hunter_env.render()
            'Hunter part'
            hunter_state = hunter_state.reshape(1, -1)
            hunter_action = hunter_agent.choose_action(hunter_state)
            # hunter_action = np.argmax(hunter_agent.target_model.predict(hunter_state))

            dist = abs(hunter_pos - warrior_pos)
            hunter_reward = 0  # min(dist, hunter_range) * 5

            if hunter_action == 0:
                hunter_reward -= 0.5
                # print('hunter stop')
            elif hunter_action == 1:
                # print('hunter move left')
                hunter_pos = max(-1, hunter_pos - hunter_mov_spd)
            elif hunter_action == 2:
                # print('hunter move right')
                hunter_pos = min(1, hunter_pos + hunter_mov_spd)
            elif hunter_action == 3:
                if hunter_env.use_skill(0, dist):
                    warrior_hp -= 20
                    hunter_reward += 2
                # print('hunter attack')
                else:
                    hunter_reward -= 1
            elif hunter_action == 4:
                if hunter_env.use_skill(1, dist):
                    hunter_reward += 4
                    warrior_hp -= 40
                # print('hunter skill')
                else:
                    hunter_reward -= 1
            elif hunter_action == 5:
                if hunter_env.use_skill(2):
                    heal = min(hunter_max_hp - hunter_hp, .2 * hunter_max_hp)
                    hunter_reward = heal / hunter_max_hp * 5
                    hunter_hp += heal
                # print('hunter heal')
            next_hunter_state, _, done, info = hunter_env.step(hunter_action)

            next_hunter_state = np.array((
                hunter_hp / hunter_max_hp,
                hunter_pos,
                warrior_hp / warrior_max_hp,
                warrior_pos,
                hunter_env.skill_timer[0] == 0,
                hunter_env.skill_timer[1] == 0,
                hunter_env.skill_timer[2] == 0
            ), dtype=np.float32)

            warrior_state = warrior_state.reshape(1, -1)
            # warrior_action = warrior_agent.choose_action(warrior_state)
            warrior_action = np.argmax(warrior_agent.target_model.predict(warrior_state))
            dist = abs(hunter_pos - warrior_pos)
            warrior_reward = -dist * .1

            if warrior_action == 0:
                warrior_reward -= 0.3
                # print('hunter stop')
            elif warrior_action == 1:
                # print('hunter move left')
                warrior_pos = max(-1, warrior_pos - warrior_mov_spd)
            elif warrior_action == 2:
                # print('hunter move right')
                warrior_pos = min(1, warrior_pos + warrior_mov_spd)
            elif warrior_action == 3:
                if warrior_env.use_skill(0, dist):
                    hunter_hp -= 16
                    warrior_reward += 1
                # print('hunter attack')
                else:
                    warrior_reward -= 1
            elif warrior_action == 4:
                if warrior_env.use_skill(1, dist):
                    warrior_pos = hunter_pos
                    warrior_reward += 1
                    hunter_hp -= 15
                # print('hunter skill')
                else:
                    warrior_reward -= 1
            elif warrior_action == 5:
                if warrior_env.use_skill(2):
                    heal = min(warrior_max_hp - warrior_hp, warrior_max_hp * .1)
                    warrior_reward = heal / warrior_max_hp * 5
                    warrior_hp += heal
                # print('hunter heal')
            next_warrior_state, _, done, info = warrior_env.step(warrior_action)

            next_warrior_state = np.array((
                warrior_hp / warrior_max_hp,
                warrior_pos,
                hunter_hp / hunter_max_hp,
                hunter_pos,
                warrior_env.skill_timer[0] == 0,
                warrior_env.skill_timer[1] == 0,
                warrior_env.skill_timer[2] == 0
            ), dtype=np.float32)

            if warrior_hp < 0:
                hunter_reward = 10  # 50 + hunter_hp/hunter_max_hp * 200
                warrior_reward += 0
            elif hunter_hp < 0:
                hunter_reward += 0
                warrior_reward = 10  # 50 + warrior_hp/warrior_max_hp * 200

            hunter_agent.remember(hunter_state, hunter_action, hunter_reward, next_hunter_state.reshape(1, -1), done)

            hunter_state = next_hunter_state
            hunter_env.state = hunter_state
            hunter_env.hp = hunter_hp
            hunter_env.pos = hunter_pos

            warrior_agent.remember(warrior_state, warrior_action, warrior_reward, next_warrior_state.reshape(1, -1),
                                   done)
            warrior_state = next_warrior_state
            warrior_env.state = warrior_state
            warrior_env.hp = warrior_hp
            warrior_env.pos = warrior_pos

            hunter_score += hunter_reward
            warrior_score += warrior_reward

            score_avg += hunter_score

            if cnt >= 300:
                done = True
            if len(hunter_agent.memory) >= hunter_agent.train_start:
                hunter_agent.train_model()

            if done:
                print('done! hunter:{0:4f} / warrior:{1:4f}'.format(hunter_score, warrior_score))
                hunter_agent.update_target_model()
                # warrior_agent.update_target_model()
                score = max(warrior_max_hp - warrior_hp, 0)
                scores.append(hunter_score)
                episodes.append(e)
                plt.plot(episodes, scores, 'b')
                plt.xlabel('episode')
                plt.ylabel('damage')
                plt.savefig('cartpole_graph.png')
                if hunter_score >= 50:
                    hunter_agent.save_weight('hunter')
                    hunter_agent.save_model('hunter')
                    # sys.exit()
