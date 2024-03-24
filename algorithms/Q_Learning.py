import numpy as np

class Q_Learning_Agents:

    def __init__(self, env,LEARNING_COUNT,TURN_LIMIT,ALPHA,GAMMA):
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


    def learn(self,epsilon):
        # one episode learning
        state = self.env.reset()

        self.episode_reward_2=0
        self.episode_reward_1=0
        state=self.env.current_game_state
        move_completed=[False,False]
        agent_0_has_finished=False
        agent_1_has_finished=False
        for t in range(self.turn_limit):

            # NEED TO CHANGE THIS SO THIS IS EPSILON GREEDY, NOT THIS RANDOM SAMPLE

            #act0,act1 = env.action_space[0].sample(),env.action_space[1].sample()
            act0,act1=self.epsilon_greedy(self.env,epsilon)
            next_state, rewards,move_completed=self.env.step([act0,act1]) 
            act0=int(act0)
            act1=int(act1)

            #act = self.env.action_space.sample() # random
            #next_state, reward, done, info = self.env.step(act, self.idx)
            assert len(next_state) == 8
            a1_state=self.env.get_state_single(list(map(int,state)),0)
            a2_state=self.env.get_state_single(list(map(int,state)),1)
            a1_next_state=self.env.get_state_single(list(map(int,next_state)),0)
            a2_next_state=self.env.get_state_single(list(map(int,next_state)),1)

            #Env state has 8 variables. Each agent's state only has 7!
            
   
            q_next_max_1=np.max(self.q_val_1[tuple(a1_next_state)])
            q_next_max_2=np.max(self.q_val_2[tuple(a2_next_state)])

            # Q <- Q + a(Q' - Q)
            # <=> Q <- (1-a)Q + a(Q')

            
            if not agent_0_has_finished:
                current_state_1=a1_state+[act0]
                #print(current_state_1)
                self.q_val_1[tuple(current_state_1)] = (1 -self.alpha) * self.q_val_1[tuple(current_state_1)]\
                                 + self.alpha * (rewards[0] + self.gamma * q_next_max_1)
            if not agent_1_has_finished:
                current_state_2=a2_state+[act1]
                self.q_val_2[tuple(current_state_2)] = (1 - self.alpha) * self.q_val_2[tuple(current_state_2)]\
                                 + self.alpha * (rewards[1] + self.gamma * q_next_max_2)
            self.episode_reward_1 += rewards[0]
            self.episode_reward_2 += rewards[1]
            #self.env.render()

            # SET FLAGS SO THAT IF AGENT HAS REACHED GOAL, Q FUNCTION WONT BE UPDATED FROM NEXT ITERAITON ONWARDS
            if move_completed[0]:
                agent_0_has_finished=True
            elif move_completed[1]:
                agent_1_has_finished=True

            if all(move_completed):
                #print(str(t+1)+" steps")
                return self.env.episode_total_reward,self.episode_reward_1,self.episode_reward_2
            else:
                state = next_state
        #print(str(t+1)+" steps")        
        return self.env.episode_total_reward,self.episode_reward_1,self.episode_reward_2
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