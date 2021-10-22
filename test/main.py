import numpy as np
import matplotlib.pyplot as plt
import os
import sys
from ml_agent import DQNAgent
from game_agent import AgentEnv
import tensorflow as tf

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
hunter_load_model = True
warrior_load_model = True
render = False

def simulation(x,step):
    # max_hp, armor, str, crit, accuracy, avoid, atk_duration, reg_hp = params
    hunter_params = (200, 0, 5, 0.05, .95, .05, 5, 0.5)
    warrior_params = (x, 0, 10, 0.05, .95, .05, 7, 0)
    hunter_env = AgentEnv("hunter", hunter_params, (10, 25), warrior_params[0])
    warrior_env = AgentEnv("warrior", warrior_params, (15, 15), hunter_params[0])

    state_size = hunter_env.observation_space.shape[0]
    action_size = hunter_env.action_space.n
    print('s_size', state_size, 'a_size', action_size)
    hunter_agent = DQNAgent(state_size, action_size)
    warrior_agent = DQNAgent(state_size, action_size)
    if hunter_load_model:
        hunter_agent.load_model('hunter')
    if warrior_load_model:
        warrior_agent.load_model('warrior')

    scores, episodes = [], []
    num_episode = step
    hunter_win = 0
    warrior_win = 0
    for e in range(num_episode):
        done = False
        warrior_state = warrior_env.reset()
        hunter_state = hunter_env.reset()
        hunter_score = 0
        warrior_score = 0
        cnt = 0
        while not done:
            cnt += 1

            'Hunter part'
            amount = 0
            hunter_state = hunter_state.reshape(1, -1)
            if hunter_load_model:
                hunter_action = np.argmax(hunter_agent.target_model.predict(hunter_state))
            else:
                hunter_action = hunter_agent.choose_action(hunter_state)
            if hunter_action == 0:
                if hunter_env.attack():
                    amount = hunter_env.give_damage(hunter_env.cal_amount(hunter_env.str))
                    warrior_env.take_damage(amount)
            elif hunter_action == 1:
                if hunter_env.use_skill(0):
                    amount = hunter_env.give_damage(hunter_env.cal_amount(hunter_env.str * 2.5))
                    warrior_env.take_damage(amount)
            elif hunter_action == 2:
                if hunter_env.use_skill(1):
                    hunter_env.add_buff({'deal_ratio': 1.5}, 10)
                    amount = hunter_env.cal_amount(hunter_env.max_hp * .2)
                    hunter_env.heal(amount)
            hunter_reward = amount/5

            amount = 0
            'Warrior part'
            warrior_state = warrior_state.reshape(1, -1)
            if warrior_load_model:
                warrior_action = np.argmax(warrior_agent.target_model.predict(warrior_state))
            else:
                warrior_action = warrior_agent.choose_action(warrior_state)
            if warrior_action == 0:
                if warrior_env.attack():
                    amount = warrior_env.give_damage(warrior_env.cal_amount(warrior_env.str))
                    hunter_env.take_damage(amount)
            elif warrior_action == 1:
                if warrior_env.use_skill(0):
                    warrior_env.add_buff({'armor': 10}, 8)
            elif warrior_action == 2:
                if warrior_env.use_skill(1):
                    hunter_env.add_buff({'heal_ratio': .5}, 10)
                    amount = warrior_env.give_damage(warrior_env.cal_amount(warrior_env.str * 2.5))
                    hunter_env.take_damage(amount)
            warrior_reward = amount/10
            warrior_env.set_enemy_hp(hunter_env.state[0])
            next_warrior_state = warrior_env.step(warrior_action)
            hunter_env.set_enemy_hp(warrior_env.state[0])
            next_hunter_state = hunter_env.step(hunter_action)


            if hunter_env.is_dead() and not warrior_env.is_dead():
                warrior_win += 1
                warrior_reward += 100
            elif not hunter_env.is_dead() and warrior_env.is_dead():
                hunter_win += 1
                hunter_reward += 100
            if hunter_env.is_dead() or warrior_env.is_dead() or cnt >= 300:
                hunter_env.state[0] = max(0, hunter_env.state[0])
                warrior_env.state[0] = max(0, warrior_env.state[0])
                next_warrior_state[0] = warrior_env.state[0]
                next_hunter_state[0] = hunter_env.state[0]
                hunter_reward += hunter_env.hp_ratio() * 50
                warrior_reward += warrior_env.hp_ratio() * 50
                done = True

            hunter_agent.remember(hunter_state, hunter_action, hunter_reward, next_hunter_state.reshape(1, -1), done)
            hunter_env.state = next_hunter_state
            hunter_score += hunter_reward

            warrior_agent.remember(warrior_state, warrior_action, warrior_reward, next_warrior_state.reshape(1, -1),
                                   done)
            warrior_env.state = next_warrior_state
            warrior_score += warrior_reward

            if render:
                hunter_env.render()
                warrior_env.render()

            if not hunter_load_model and len(hunter_agent.memory) >= hunter_agent.train_start:
                hunter_agent.train_model()
            if not warrior_load_model and len(warrior_agent.memory) >= warrior_agent.train_start:
                warrior_agent.train_model()
            if done:
                print(hunter_score)
                print('[{0:03d}] hunter:{1:4f} warrior:{2:4f}'.format(e, hunter_env.state[0], warrior_env.state[0]))
                scores.append(warrior_env.state[0] - hunter_env.state[0])
                if not hunter_load_model:
                    hunter_agent.update_target_model()
                if not warrior_load_model:
                    warrior_agent.update_target_model()
    if not hunter_load_model:
        hunter_agent.save_weight('hunter')
        hunter_agent.save_model('hunter')
    if not warrior_load_model:
        warrior_agent.save_weight('warrior')
        warrior_agent.save_model('warrior')
    draw = num_episode - hunter_win - warrior_win
    game_result = (hunter_win, warrior_win, draw)
    if draw == num_episode:
        win_rate = .5
    else:
        win_rate = hunter_win / (num_episode - draw)
    mean, mse, var, std = np.mean(scores), np.square(scores).mean(), np.var(scores), np.std(scores)

    print('win_rate:{0:.5f}/mean:{1:.5f}/mse:{2:.5f}/var:{3:.5f}/std:{4:.5f}'.format(win_rate, mean, mse, var, std))
    print('game_result:',game_result)
    return mean, mse, win_rate

hunter_load_model = False
simulation(200,100)
hunter_load_model = True
warrior_load_model = False
simulation(200,100)
warrior_load_model = True

if __name__ == "__main__":
    x= init_x = 300
    episodes, means, win_rates, parameters = [],[],[],[]
    num = 100
    for i in range(num):
        print('[{0:03d}] parameter:{1}'.format(i,x))
        mean, mse, win_rate = simulation(x,10)
        x -= mean *(num-i)/num / 10
        episodes.append(i)
        means.append(mse)
        parameters.append(x)
        win_rates.append(win_rate)


    plt.plot(episodes, means, 'b')
    plt.xlabel('episode')
    plt.ylabel('mean square error')
    plt.savefig('simulation_graph.png')
    plt.clf()
    plt.plot(episodes, win_rates, 'r')
    plt.xlabel('episode')
    plt.ylabel('win_rate')
    plt.savefig('win_rate_graph.png')

    plt.clf()
    plt.plot(episodes, parameters, 'g')
    plt.xlabel('episode')
    plt.ylabel('parameter')
    plt.savefig('parameter_graph.png')
    simulation(init_x, 50)
    simulation(x, 50)
