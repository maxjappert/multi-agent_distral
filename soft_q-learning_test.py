import matplotlib.pyplot as plt
import numpy as np
from gridworld_env import GridworldEnv


# Q learning params
ALPHA = 0.2 # learning rate
GAMMA = 0.95 # reward discount
TAU=0.5
LEARNING_COUNT = 1000
TEST_COUNT = 200

NUM_GAMES = 50

TURN_LIMIT = 200
IS_MONITOR = True
from algorithms.Soft_Q_Learning_without_rollout import Soft_without_rollout
#from new_test import SoftWithoutRolloutMC

env = GridworldEnv('6')
env.reset()
agents = Soft_without_rollout(env,LEARNING_COUNT,TURN_LIMIT,ALPHA,GAMMA,TAU)
#agents = Soft_without_rollout(env, LEARNING_COUNT, TURN_LIMIT, GAMMA)

print("###### LEARNING #####")
reward_total_1 = 0.0
reward_total_2 = 0.0
rewards_1 = []
rewards_2 = []
total_rewards = []
for i in range(LEARNING_COUNT):

    if (i+1) % 10 == 0:
        print(f'Epoch {i+1}')

    epoch_reward_1 = 0
    epoch_reward_2 = 0
    for j in range(NUM_GAMES):
        total_reward,reward_1,reward_2=agents.learn(i)
        reward_total_1 += reward_1
        reward_total_2 += reward_2
        epoch_reward_1 += reward_1
        epoch_reward_2 += reward_2
    rewards_1.append(epoch_reward_1 / NUM_GAMES)
    rewards_2.append(epoch_reward_2 / NUM_GAMES)

    if (i+1) % 100 == 0:
        plt.figure()
        plt.title(f'Epoch {i+1}')
        plt.plot(range(len(rewards_1)), rewards_1, label='Agent 1')
        plt.plot(range(len(rewards_2)), rewards_2, label='Agent 2')
        plt.legend()
        plt.show()

print("episodes      : {}".format(LEARNING_COUNT))
print("agent 1 total reward  : {}".format(reward_total_1))
print("agent 1 average reward: {:.2f}".format(reward_total_1 / LEARNING_COUNT))
#print("agent 1 Q Value       :{}".format(agent.q_val))
print("agent 2 total reward  : {}".format(reward_total_2))
print("agent 2 average reward: {:.2f}".format(reward_total_2 / LEARNING_COUNT))

