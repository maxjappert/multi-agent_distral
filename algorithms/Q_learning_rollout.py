import numpy as np

class Q_Learning_Rollout_Agents:

    def __init__(self, env,LEARNING_COUNT,TURN_LIMIT,ALPHA,GAMMA,rollout):
        self.env = env
        self.episode_reward_1 = 0.0
        self.episode_reward_2 = 0.0
        #remember the actions in indices 2 and 6 are the previous actions, but this is simply part of the state (i.e. not the "a" in Q(s,a))
        # need to add another dim at the end to actually select next action

        #Note that env state has all 8 variables, but each agent's state will only have 6 (discard the other agent's previous action and own agent's reward)
        self.q_val_1 = np.zeros(7*9*7*9*5*2*5).reshape(7,9,7,9,5,2,5).astype(np.float32)
        self.q_val_2 = np.zeros(7*9*7*9*5*2*5).reshape(7,9,7,9,5,2,5).astype(np.float32)
        self.lerning_count=LEARNING_COUNT
        self.turn_limit=TURN_LIMIT
        self.alpha=ALPHA 
        self.gamma=GAMMA
        self.rollout=rollout

    def update_epsilon(self,episode, min_epsilon=0.01, max_epsilon=1.0, decay_rate=0.01):
        """
        Updates epsilon using an exponential decay formula.
        
        Parameters:
            episode (int): The current episode number.
            total_episodes (int): The total number of episodes.
            min_epsilon (float): The minimum value epsilon can take.
            max_epsilon (float): The maximum value epsilon can start from.
            decay_rate (float): The rate at which epsilon decays over time.
            
        Returns:
            float: The updated epsilon value.
        """
        epsilon = min_epsilon + (max_epsilon - min_epsilon) * np.exp(-decay_rate * episode)
        return epsilon

    def epsilon_greedy(self,env,epsilon):
        random_numbers=np.random.rand(2)
        a1_state=self.env.get_state_single(list(map(int,env.current_game_state)),0)
        a2_state=self.env.get_state_single(list(map(int,env.current_game_state)),1)
        if random_numbers[0]<epsilon:
            act1=env.action_space[0].sample()
        else:
            act1=np.argmax(self.q_val_1[tuple(a1_state)])
        
        if random_numbers[1]<epsilon:
            act2=env.action_space[1].sample()
        else:
            act2=np.argmax(self.q_val_2[tuple(a2_state)])

        return act1,act2


    def learn(self, epsilon,record=False):
        state = self.env.reset()

        self.episode_reward_2 = 0
        self.episode_reward_1 = 0
        state = self.env.current_game_state
        if record:
            self.env.render(mode='write')
        move_completed = [False, False]
        agent_0_has_finished = False
        agent_1_has_finished = False

        states_1 = [self.env.get_state_single(list(map(int, state)), 0)]
        states_2 = [self.env.get_state_single(list(map(int, state)), 1)]
        actions_1 = []
        actions_2 = []
        rewards_1 = []
        rewards_2 = []

        act0, act1 = self.epsilon_greedy(self.env, epsilon)
        actions_1.append(act0)
        actions_2.append(act1)
        next_state, next_rewards, move_completed = self.env.step([act0, act1])
        rewards=[0,0]
        rewards_1.append(rewards[0])
        rewards_2.append(rewards[1])

        state=next_state
        rewards=next_rewards
        if record:
            self.env.render(mode='write')


        for t in range(self.turn_limit):
            act0, act1 = self.epsilon_greedy(self.env, epsilon)
            next_state, next_rewards, move_completed = self.env.step([act0, act1])
            if record:
                self.env.render('write')
            act0 = int(act0)
            act1 = int(act1)

            a1_state = self.env.get_state_single(list(map(int, state)), 0)
            a2_state = self.env.get_state_single(list(map(int, state)), 1)
            a1_next_state = self.env.get_state_single(list(map(int, next_state)), 0)
            a2_next_state = self.env.get_state_single(list(map(int, next_state)), 1)

            states_1.append(a1_state)
            states_2.append(a2_state)
            actions_1.append(act0)
            actions_2.append(act1)
            rewards_1.append(rewards[0])
            rewards_2.append(rewards[1])

            self.episode_reward_1 += rewards[0]
            self.episode_reward_2 += rewards[1]

            if all(move_completed) or len(states_1) >= 10:
                if all(move_completed) and all(reward==0 for reward in rewards):
                    return self.env.episode_total_reward, self.episode_reward_1, self.episode_reward_2
                # Rollout update for agent 1
                for t in range(len(states_1) - 1):
                    state_1 = states_1[t]
                    action_1 = actions_1[t]
                    reward_1 = rewards_1[t]
                    next_state_1 = states_1[t + 1]

                    q_next_max_1 = np.max(self.q_val_1[tuple(next_state_1)])
                    td_target = reward_1 + self.gamma * q_next_max_1
                    td_error = td_target - self.q_val_1[tuple(state_1 + [action_1])]
                    self.q_val_1[tuple(state_1 + [action_1])] += self.alpha * td_error

                # Rollout update for agent 2
                for t in range(len(states_2) - 1):
                    state_2 = states_2[t]
                    action_2 = actions_2[t]
                    reward_2 = rewards_2[t]
                    next_state_2 = states_2[t + 1]

                    q_next_max_2 = np.max(self.q_val_2[tuple(next_state_2)])
                    td_target = reward_2 + self.gamma * q_next_max_2
                    td_error = td_target - self.q_val_2[tuple(state_2 + [action_2])]
                    self.q_val_2[tuple(state_2 + [action_2])] += self.alpha * td_error

                states_1 = []
                states_2 = []
                actions_1 = []
                actions_2 = []
                rewards_1 = []
                rewards_2 = []
                state = next_state
                rewards=next_rewards
            else:
                state = next_state
                rewards=next_rewards
        return self.env.episode_total_reward, self.episode_reward_1, self.episode_reward_2
    """
    def test(self):
        state = self.env.reset()
        for t in range(TURN_LIMIT):
            norm_coords = state[self.idx * 3]
            act = np.argmax(self.q_val[int(64.*(norm_coords+1.)/2.)])
            next_state, reward, done, info = self.env.step(act, self.idx)
            if done:
                return self.env.episode_total_reward
            else:
                state = next_state
        return 0.0 # over limit
    """