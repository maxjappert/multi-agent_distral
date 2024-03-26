import numpy as np
from scipy.special import logsumexp
class Soft_without_rollout:

    def __init__(self, env,LEARNING_COUNT,TURN_LIMIT,ALPHA,GAMMA,TAU):
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
        self.tau=TAU

    def softmax_action_selection(self, env, state, agent_id):
        if agent_id == 0:
            q_values = self.q_val_1[tuple(state)]
            log_action_probs = (q_values / self.tau)-logsumexp((q_values / self.tau))
            #action_probs=np.nan_to_num(action_probs)
            #action_probs=action_probs/np.sum(action_probs)
            #print(log_action_probs)
            #probs=np.exp(log_action_probs)/np.sum(np.exp(log_action_probs))
            #print(probs)
            action = np.random.choice(env.action_space[0].n, p=np.exp(log_action_probs))
        else:
            q_values = self.q_val_2[tuple(state)]
            log_action_probs = (q_values / self.tau)-logsumexp((q_values / self.tau))
            #action_probs=np.nan_to_num(action_probs)
            #action_probs=action_probs/np.sum(action_probs)
            #print(log_action_probs)
            #probs=np.exp(log_action_probs)/np.sum(np.exp(log_action_probs))
            #print(probs)
            action = np.random.choice(env.action_space[0].n, p=np.exp(log_action_probs))
        return action


    def learn(self,record=False):
        # one episode learning
        state = self.env.reset()

        self.episode_reward_2=0
        self.episode_reward_1=0
        state=self.env.current_game_state
        move_completed=[False,False]
        #agent_0_has_finished=False
        #agent_1_has_finished=False
        act0=self.softmax_action_selection(self.env,self.env.get_state_single(list(map(int,state)),0),0)
        act1=self.softmax_action_selection(self.env,self.env.get_state_single(list(map(int,state)),1),1)
        if record:
            self.env.render(mode='write')
        next_state, next_rewards,move_completed=self.env.step([act0,act1]) 
        rewards=[0,0]
        assert len(next_state) == 8
        a1_state=self.env.get_state_single(list(map(int,state)),0)
        a2_state=self.env.get_state_single(list(map(int,state)),1)
        a1_next_state=self.env.get_state_single(list(map(int,next_state)),0)
        a2_next_state=self.env.get_state_single(list(map(int,next_state)),1)
        q_next_max_1=np.max(self.q_val_1[tuple(a1_next_state)])
        q_next_max_2=np.max(self.q_val_2[tuple(a2_next_state)])
        
        act0=int(act0)
        act1=int(act1)
        current_state_1=a1_state+[act0]
        
        q_next_max_1 = self.tau * np.log(np.sum(np.exp(self.q_val_1[tuple(a1_next_state)] / self.tau)))
        td_target = rewards[0] + self.gamma * q_next_max_1
        td_error = td_target - self.q_val_1[tuple(current_state_1)]
        self.q_val_1[tuple(current_state_1)] += self.alpha * td_error

        current_state_2=a2_state+[act1]

        
        q_next_max_2 = self.tau * np.log(np.sum(np.exp(self.q_val_2[tuple(a2_next_state)] / self.tau)))
        td_target = rewards[1] + self.gamma * q_next_max_2
        td_error = td_target - self.q_val_2[tuple(current_state_2)]
        self.q_val_2[tuple(current_state_2)] += self.alpha * td_error

        state=next_state
        rewards=next_rewards
        if record:
            self.env.render(mode='write')
        for t in range(self.turn_limit):
            if all(move_completed) and all(reward == 0 for reward in rewards):
                return self.env.episode_total_reward,self.episode_reward_1,self.episode_reward_2

            self.episode_reward_1 += rewards[0]
            self.episode_reward_2 += rewards[1]

            
            
            act0=self.softmax_action_selection(self.env,self.env.get_state_single(list(map(int,state)),0),0)
            act1=self.softmax_action_selection(self.env,self.env.get_state_single(list(map(int,state)),1),1)
            next_state, next_rewards,move_completed=self.env.step([act0,act1]) 
            if record:
                self.env.render(mode='write')


            assert len(next_state) == 8
            a1_state=self.env.get_state_single(list(map(int,state)),0)
            a2_state=self.env.get_state_single(list(map(int,state)),1)
            a1_next_state=self.env.get_state_single(list(map(int,next_state)),0)
            a2_next_state=self.env.get_state_single(list(map(int,next_state)),1)

            #Env state has 8 variables. Each agent's state only has 7!
            
            #q_next_max_1=np.max(self.q_val_1[tuple(a1_next_state)])
            #q_next_max_2=np.max(self.q_val_2[tuple(a2_next_state)])

            act0=int(act0)
            act1=int(act1)
            flag=False
            if not (move_completed[0] and rewards[0]==0.0) :
                current_state_1=a1_state+[act0]
                flag=True
                #print(current_state_1)
                q_next_max_1 = self.tau * np.log(np.sum(np.exp(self.q_val_1[tuple(a1_next_state)] / self.tau)))
                td_target = rewards[0] + self.gamma * q_next_max_1
                td_error = td_target - self.q_val_1[tuple(current_state_1)]
                self.q_val_1[tuple(current_state_1)] += self.alpha * td_error
            if not (move_completed[1] and rewards[1]==0.0):
                current_state_2=a2_state+[act1]
                flag=True
                q_next_max_2 = self.tau * np.log(np.sum(np.exp(self.q_val_2[tuple(a2_next_state)] / self.tau)))
                td_target = rewards[1] + self.gamma * q_next_max_2
                td_error = td_target - self.q_val_2[tuple(current_state_2)]
                self.q_val_2[tuple(current_state_2)] += self.alpha * td_error
            #if not flag:
            #   print(rewards)
            state=next_state
            rewards=next_rewards

        
        #print(str(t+1)+" steps")        
        return self.env.episode_total_reward,self.episode_reward_1,self.episode_reward_2