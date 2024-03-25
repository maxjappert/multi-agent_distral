import numpy as np
import math

class MultiDistral_v1:
    def __init__(self, env, LEARNING_COUNT, TURN_LIMIT, ALPHA, GAMMA, BETA, TAU):
        self.env = env
        self.episode_reward_1 = 0.0
        self.episode_reward_2 = 0.0
        self.q_val_1 = np.zeros(7*9*7*9*5*2*5, dtype=np.float32)
        self.q_val_2 = np.zeros(7*9*7*9*5*2*5, dtype=np.float32)
        self.v_val_1 = np.zeros(7*9*7*9*5*2, dtype=np.float32)
        self.v_val_2 = np.zeros(7*9*7*9*5*2, dtype=np.float32)
        self.pi_0 = np.ones(7*9*7*9*5*2*5, dtype=np.float32) / (7*9*7*9*5*2*5)  # Initialize uniform prior policy
        self.learning_count = LEARNING_COUNT
        self.turn_limit = TURN_LIMIT
        self.alpha = ALPHA
        self.gamma = GAMMA
        self.beta = BETA  # Inverse temperature parameter
        self.tau = TAU

    def softmax_action_selection(self, env, state, agent_id):
        if agent_id == 0:
            q_values = self.q_val_1[np.ravel_multi_index(state, self.q_val_1.shape)]
            v_value_shape = self.v_val_1.shape
            v_value = self.v_val_1[np.ravel_multi_index(state[:len(v_value_shape)], v_value_shape)]
            advantage = q_values - v_value
            action_probs = self.pi_0[np.ravel_multi_index(state, self.pi_0.shape)] * np.exp(self.beta * advantage)
            action_probs /= action_probs.sum()
            action = np.random.choice(env.action_space[0].n, p=action_probs)
        else:
            q_values = self.q_val_2[np.ravel_multi_index(state, self.q_val_2.shape)]
            v_value_shape = self.v_val_2.shape
            v_value = self.v_val_2[np.ravel_multi_index(state[:len(v_value_shape)], v_value_shape)]
            advantage = q_values - v_value
            action_probs = self.pi_0[np.ravel_multi_index(state, self.pi_0.shape)] * np.exp(self.beta * advantage)
            action_probs /= action_probs.sum()
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

        states_1 = [np.array(self.env.get_state_single(list(map(int, state)), 0), dtype=np.int32)]
        states_2 = [np.array(self.env.get_state_single(list(map(int, state)), 1), dtype=np.int32)]
        actions_1 = []
        actions_2 = []
        rewards_1 = []
        rewards_2 = []

        for t in range(self.turn_limit):
            act0 = self.softmax_action_selection(self.env, states_1[-1], 0)
            act1 = self.softmax_action_selection(self.env, states_2[-1], 1)
            next_state, rewards, move_completed = self.env.step([act0, act1])

            a1_state = np.array(self.env.get_state_single(list(map(int, state)), 0), dtype=np.int32)
            a2_state = np.array(self.env.get_state_single(list(map(int, state)), 1), dtype=np.int32)
            a1_next_state = np.array(self.env.get_state_single(list(map(int, next_state)), 0), dtype=np.int32)
            a2_next_state = np.array(self.env.get_state_single(list(map(int, next_state)), 1), dtype=np.int32)

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
                    state_idx_1 = np.ravel_multi_index(states_1[t], self.q_val_1.shape)
                    action_idx_1 = state_idx_1 * self.env.action_space[0].n + actions_1[t]
                    reward_1 = rewards_1[t]
                    next_state_idx_1 = np.ravel_multi_index(states_1[t+1], self.v_val_1.shape)
                    v_next_state_1 = self.v_val_1[next_state_idx_1]
                    td_target = reward_1 + self.gamma * v_next_state_1
                    td_error = td_target - self.q_val_1[action_idx_1]
                    self.q_val_1[action_idx_1] += self.alpha * td_error
                    self.v_val_1[next_state_idx_1] = np.log(np.sum(np.exp(self.q_val_1[np.ravel_multi_index(states_1[t+1], self.q_val_1.shape)] / self.tau))) / self.beta

                # Rollout update for agent 2
                for t in range(len(states_2) - 1):
                    state_idx_2 = np.ravel_multi_index(states_2[t], self.q_val_2.shape)
                    action_idx_2 = state_idx_2 * self.env.action_space[1].n + actions_2[t]
                    reward_2 = rewards_2[t]
                    next_state_idx_2 = np.ravel_multi_index(states_2[t+1], self.v_val_2.shape)
                    v_next_state_2 = self.v_val_2[next_state_idx_2]
                    td_target = reward_2 + self.gamma * v_next_state_2
                    td_error = td_target - self.q_val_2[action_idx_2]
                    self.q_val_2[action_idx_2] += self.alpha * td_error
                    self.v_val_2[next_state_idx_2] = np.log(np.sum(np.exp(self.q_val_2[np.ravel_multi_index(states_2[t+1], self.q_val_2.shape)] / self.tau))) / self.beta

                # Update prior policy pi_0 based on state-action visitation frequencies
                state_action_counts_1 = np.zeros_like(self.pi_0)
                for t in range(len(states_1) - 1):
                    state_action_idx_1 = np.ravel_multi_index(states_1[t] + [actions_1[t]], self.pi_0.shape)
                    state_action_counts_1[state_action_idx_1] += 1
                state_action_counts_2 = np.zeros_like(self.pi_0)
                for t in range(len(states_2) - 1):
                    state_action_idx_2 = np.ravel_multi_index(states_2[t] + [actions_2[t]], self.pi_0.shape)
                    state_action_counts_2[state_action_idx_2] += 1
                self.pi_0 = (state_action_counts_1 + state_action_counts_2) / (state_action_counts_1 + state_action_counts_2).sum()

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
