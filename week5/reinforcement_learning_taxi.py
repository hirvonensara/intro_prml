""" 
Introduction to Pattern Recognition and Machine Learning
Sara Hirvonen
Exercise 5 Reinforcement learning (OpenAI Gym)

"""
import gym
import random
import numpy as np
import time

def train(alpha, gamma, episodes, steps, env):
    Q_reward = np.zeros((500,6))

    # Training w/ random sampling of actions
    for i in range(episodes):
        state1 = env.reset()
        for j in range(steps):
            random_num = random.randint(0,5)
            state2, reward, done, info = env.step(random_num)
            new_value = Q_reward[state1, random_num]+alpha*(reward + gamma*max(Q_reward[state2, :])-Q_reward[state1, random_num])
            Q_reward[state1, random_num] = new_value
            state1 = state2

            if done:
                break

    return Q_reward

def test(Q_reward, env):
    tot_actions = 0
    tot_reward = 0

    state = env.reset()
    for t in range(50):
        action = np.argmax(Q_reward[state, :])
        state, reward, done, info = env.step(action)
        tot_reward += reward
        tot_actions += 1
        env.render()
        time.sleep(1)
        if done:
            print("Total reward {}".format(tot_reward))
            break

    return tot_actions, tot_reward

def main():
    # Environment
    env = gym.make("Taxi-v3")

    # Training parameters for Q learning
    alpha = 0.9 # learning rate
    gamma = 0.9 # Future reward discount factor
    num_of_episodes = 1000
    num_of_steps = 500 # per each episode

    # Q tables for rewards 
    Q_reward = train(alpha, gamma, num_of_episodes, num_of_steps, env)
    
    tot_reward = 0
    tot_actions = 0

    for _ in range(10):
        actions, reward = test(Q_reward, env)
        tot_actions += actions
        tot_reward += reward

    avg_of_actions = tot_actions/10
    avg_of_total_reward = tot_reward/10

    print("Average total reward is", avg_of_total_reward)
    print("Average number of actions is", avg_of_actions)


main()