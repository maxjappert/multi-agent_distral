# Soft Q-Learning with 10 step rollout. This will be the baseline we'll compare MultiDistral to.
# We will apply this soft q-learning to each agent, but without any shared policy.

import numpy as np
import math

class Soft_Q_Learning_Baseline_Agents_fast:
    def __init__(self, env, LEARNING_COUNT, TURN_LIMIT, ALPHA, GAMMA, TAU):
        self.env = env
        self.episode_reward_1 = 0.0
        self.episode_reward_2 = 0.0
        self.q_val_1 = np.zeros(7*9*7*9*5*2*5).reshape(7,9,7,9,5,2,5).astype(np.float32)
        self.q_val_2 = np.zeros(7*9*7*9*5*2*5).reshape(7,9,7,9,5,2,5).astype(np.float32)
        self.learning_count = LEARNING_COUNT
        self.turn_limit = TURN_LIMIT
        self.alpha = ALPHA
        self.gamma = GAMMA
        self.tau = TAU

    def softmax_action_selection(self, env, state, agent_id):
        if agent_id == 0:
            q_values = self.q_val_1[tuple(state)]
            action_probs = np.exp(q_values / self.tau) / np.sum(np.exp(q_values / self.tau))
            action = np.random.choice(env.action_space[0].n, p=action_probs)
        else:
            q_values = self.q_val_2[tuple(state)]
            action_probs = np.exp(q_values / self.tau) / np.sum(np.exp(q_values / self.tau))
            action = np.random.choice(env.action_space[1].n, p=action_probs)
        return action

    def learn(self):
        state = self.env.reset()
        self.episode_reward_2 = 0
        self.episode_reward_1 = 0
        state = self.env.current_game_state
        move_completed = [False, False]
        agent_0_has_finished = False
        agent_1_has_finished = False

        states_1 = [self.env.get_state_single(list(map(int, state)), 0)]
        states_2 = [self.env.get_state_single(list(map(int, state)), 1)]
        actions_1 = []
        actions_2 = []
        rewards_1 = []
        rewards_2 = []

        for t in range(self.turn_limit):
            act0 = self.softmax_action_selection(self.env, states_1[-1], 0)
            act1 = self.softmax_action_selection(self.env, states_2[-1], 1)
            next_state, rewards, move_completed = self.env.step([act0, act1])

            a1_state = self.env.get_state_single(list(map(int, state)), 0)
            a2_state = self.env.get_state_single(list(map(int, state)), 1)
            a1_next_state = self.env.get_state_single(list(map(int, next_state)), 0)
            a2_next_state = self.env.get_state_single(list(map(int, next_state)), 1)

            actions_1.append(act0)
            actions_2.append(act1)
            rewards_1.append(rewards[0])
            rewards_2.append(rewards[1])

            self.episode_reward_1 += rewards[0]
            self.episode_reward_2 += rewards[1]

            if move_completed[0]:
                agent_0_has_finished = True
            elif move_completed[1]:
                agent_1_has_finished = True

            if all(move_completed) or len(states_1) >= 10:
                # Rollout update for agent 1
                for t in range(len(states_1) - 1):
                    state_1 = states_1[t]
                    action_1 = actions_1[t]
                    reward_1 = rewards_1[t]
                    next_state_1 = states_1[t + 1]

                    q_next_max_1 = self.tau * np.log(np.sum(np.exp(self.q_val_1[tuple(next_state_1)] / self.tau)))
                    td_target = reward_1 + self.gamma * q_next_max_1
                    td_error = td_target - self.q_val_1[tuple(state_1 + [action_1])]
                    self.q_val_1[tuple(state_1 + [action_1])] += self.alpha * td_error

                # Rollout update for agent 2
                for t in range(len(states_2) - 1):
                    state_2 = states_2[t]
                    action_2 = actions_2[t]
                    reward_2 = rewards_2[t]
                    next_state_2 = states_2[t + 1]

                    q_next_max_2 = self.tau * np.log(np.sum(np.exp(self.q_val_2[tuple(next_state_2)] / self.tau)))
                    td_target = reward_2 + self.gamma * q_next_max_2
                    td_error = td_target - self.q_val_2[tuple(state_2 + [action_2])]
                    self.q_val_2[tuple(state_2 + [action_2])] += self.alpha * td_error

                states_1 = [a1_next_state]
                states_2 = [a2_next_state]
                actions_1 = []
                actions_2 = []
                rewards_1 = []
                rewards_2 = []

                if all(move_completed):
                    return self.env.episode_total_reward, self.episode_reward_1, self.episode_reward_2
            else:
                states_1.append(a1_next_state)
                states_2.append(a2_next_state)
                state = next_state

        return self.env.episode_total_reward, self.episode_reward_1, self.episode_reward_2