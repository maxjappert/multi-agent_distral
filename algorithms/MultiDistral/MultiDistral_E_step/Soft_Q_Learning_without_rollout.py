import numpy as np
from scipy.special import logsumexp


class Soft_without_rollout:

    def __init__(self, env,LEARNING_COUNT,TURN_LIMIT,ALPHA,GAMMA,TAU,PSI,BETA,MultiDistral_version,pi01,pi02):
        self.env = env
        self.episode_reward_1 = 0.0
        self.episode_reward_2 = 0.0
        #remember the actions in indices 2 and 6 are the previous actions, but this is simply part of the state (i.e. not the "a" in Q(s,a))
        # need to add another dim at the end to actually select next action

        #Note that env state has all 8 variables, but each agent's state will only have 6 (discard the other agent's previous action and own agent's reward)
        self.q_val_1 = np.zeros(7*9*7*9*5*2*5).reshape(7,9,7,9,5,2,5).astype(np.float32)
        self.v_val_1 = np.zeros(7*9*7*9*5*2).reshape(7,9,7,9,5,2).astype(np.float32)
        self.q_val_2 = np.zeros(7*9*7*9*5*2*5).reshape(7,9,7,9,5,2,5).astype(np.float32)
        self.v_val_2 = np.zeros(7*9*7*9*5*2).reshape(7,9,7,9,5,2).astype(np.float32)
        self.pi1 = np.ones(7*9*7*9*5*2*5).reshape(7,9,7,9,5,2,5).astype(np.float32)/5
        self.pi2 = np.ones(7*9*7*9*5*2*5).reshape(7,9,7,9,5,2,5).astype(np.float32) /5
        self.lerning_count=LEARNING_COUNT
        self.turn_limit=TURN_LIMIT
        self.alpha=ALPHA 
        self.gamma=GAMMA
        self.tau=TAU
        self.version=MultiDistral_version
        self.pi_0_1=pi01
        self.pi_0_2=pi02


        # THIS IS ALPHA IN THE ORIGINAL PAPER, TO DESIGNATE C_kl/c_kl+c_ent (here not called alpha cause that is the learning rate)
        self.psi=PSI

        # BETA AS IN THE PAPER
        self.beta=BETA
    
    # Function to find/act according to pi_i
    def action_selection(self, env, state, agent_id):
        if agent_id == 0:
            q_values = self.q_val_1[tuple(state)]
            v_value=self.v_val_1[tuple(state)]
            advantages=q_values-v_value
            log_action_probs = self.psi*np.log(self.pi_0_1[tuple(state)])+(self.beta*advantages)
            probs=np.exp(log_action_probs)/np.sum(np.exp(log_action_probs))
            #print(probs)
            action = np.random.choice(env.action_space[0].n, p=probs)
            #print(self.pi1[tuple(state)])
            self.pi1[tuple(state)]=probs
        else:
            q_values = self.q_val_2[tuple(state)]
            v_value=self.v_val_2[tuple(state)]
            advantages=q_values-v_value
            log_action_probs = self.psi*np.log(self.pi_0_2[tuple(state)])+(self.beta*advantages)
            probs=np.exp(log_action_probs)/np.sum(np.exp(log_action_probs))
            #print(probs)
            action = np.random.choice(env.action_space[1].n, p=probs)
            #print(self.pi2[tuple(state)])
            self.pi2[tuple(state)]=probs
        return action,np.exp(log_action_probs)


    def learn(self,record=False):
        # one episode learning
        state = self.env.reset()
        if self.version==2:
            self.pi_0_2=self.pi_0_1


        self.episode_reward_2=0
        self.episode_reward_1=0
        state=self.env.current_game_state
        move_completed=[False,False]
        #agent_0_has_finished=False
        #agent_1_has_finished=False
        act0,_=self.action_selection(self.env,self.env.get_state_single(list(map(int,state)),0),0)
        act1,_=self.action_selection(self.env,self.env.get_state_single(list(map(int,state)),1),1)
        if record:
            self.env.render(mode='write')
        next_state, next_rewards,move_completed=self.env.step([act0,act1]) 
        rewards=[0,0]
        assert len(next_state) == 8
        a1_state=self.env.get_state_single(list(map(int,state)),0)
        a2_state=self.env.get_state_single(list(map(int,state)),1)
        a1_next_state=self.env.get_state_single(list(map(int,next_state)),0)
        a2_next_state=self.env.get_state_single(list(map(int,next_state)),1)

        
        act0=int(act0)
        act1=int(act1)
        current_state_1=a1_state+[act0]
        

        # EQ (3) IN PAPER BUT MODEL FREE
        td_target = rewards[0] + self.gamma * self.v_val_1[tuple(a1_next_state)]
        td_error = td_target - self.q_val_1[tuple(current_state_1)]
        self.q_val_1[tuple(current_state_1)] += self.alpha * td_error

        # EQ (2) IN PAPER
        exp_q_val = np.exp(self.beta * self.q_val_1[tuple(a1_state)])
        # Compute ∑_at π_α^0(at|st) exp[βQi(at, st)]
        weighted_exp_q_val = np.multiply(self.pi_0_1[tuple(a1_state)],exp_q_val)
        weighted_exp_q_val_sum = weighted_exp_q_val.sum(axis=-1)
        #print(weighted_exp_q_val_sum)
        #print(self.v_val_1[tuple(a1_state)])
        self.v_val_1[tuple(a1_state)]=1/self.beta * np.log(weighted_exp_q_val_sum)

        current_state_2=a2_state+[act1]
        # EQ (3) IN PAPER BUT MODEL FREE
        td_target = rewards[1] + self.gamma * self.v_val_2[tuple(a2_next_state)]
        td_error = td_target - self.q_val_2[tuple(current_state_2)]
        self.q_val_2[tuple(current_state_2)] += self.alpha * td_error

        # EQ (2) IN PAPER
        exp_q_val = np.exp(self.beta * self.q_val_2[tuple(a2_state)])
        # Compute ∑_at π_α^0(at|st) exp[βQi(at, st)]
        weighted_exp_q_val = np.multiply(self.pi_0_2[tuple(a2_state)], exp_q_val)
        weighted_exp_q_val_sum = weighted_exp_q_val.sum(axis=-1)
        self.v_val_2[tuple(a2_state)]=1/self.beta * np.log(weighted_exp_q_val_sum)

        state=next_state
        rewards=next_rewards
        if record:
            self.env.render(mode='write')
        for t in range(self.turn_limit):
            if all(move_completed) and all(reward == 0 for reward in rewards):
                return self.env.episode_total_reward,self.episode_reward_1,self.episode_reward_2

            self.episode_reward_1 += rewards[0]
            self.episode_reward_2 += rewards[1]

            
            
            act0,_=self.action_selection(self.env,self.env.get_state_single(list(map(int,state)),0),0)
            act1,_=self.action_selection(self.env,self.env.get_state_single(list(map(int,state)),1),1)
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
                # EQ (3) IN PAPER BUT MODEL FREE
                td_target = rewards[0] + self.gamma * self.v_val_1[tuple(a1_next_state)]
                td_error = td_target - self.q_val_1[tuple(current_state_1)]
                self.q_val_1[tuple(current_state_1)] += self.alpha * td_error

                # EQ (2) IN PAPER
                exp_q_val = np.exp(self.beta * self.q_val_1[tuple(a1_state)])
                # Compute ∑_at π_α^0(at|st) exp[βQi(at, st)]
                weighted_exp_q_val = np.multiply(self.pi_0_1[tuple(a1_state)],exp_q_val)
                weighted_exp_q_val_sum = weighted_exp_q_val.sum(axis=-1)
                self.v_val_1[tuple(a1_state)]=1/self.beta * np.log(weighted_exp_q_val_sum)
            if not (move_completed[1] and rewards[1]==0.0):
                # EQ (3) IN PAPER BUT MODEL FREE
                td_target = rewards[1] + self.gamma * self.v_val_2[tuple(a2_next_state)]
                td_error = td_target - self.q_val_2[tuple(current_state_2)]
                self.q_val_2[tuple(current_state_2)] += self.alpha * td_error

                # EQ (2) IN PAPER
                exp_q_val = np.exp(self.beta * self.q_val_2[tuple(a2_state)])
                # Compute ∑_at π_α^0(at|st) exp[βQi(at, st)]
                weighted_exp_q_val = np.multiply(self.pi_0_2[tuple(a2_state)],exp_q_val)
                weighted_exp_q_val_sum = weighted_exp_q_val.sum(axis=-1)
                self.v_val_2[tuple(a2_state)]=1/self.beta * np.log(weighted_exp_q_val_sum)
            #if not flag:
            #   print(rewards)
            state=next_state
            rewards=next_rewards

        
        #print(str(t+1)+" steps")        
        return self.env.episode_total_reward,self.episode_reward_1,self.episode_reward_2

    def M_step_action_selection(self,env,state1,state2,pi_i_1,pi_i_2):
        print(pi_i_1[tuple(state1)])
        print(pi_i_2[tuple(state2)])
        action1 = np.random.choice(env.action_space[0].n, p=pi_i_1[tuple(state1)])
        action2 = np.random.choice(env.action_space[1].n, p=pi_i_2[tuple(state2)])
        return action1,action2
    
    def M_step(self,pi_i_1,pi_1_2):
        state = self.env.reset()
        state=self.env.current_game_state
        move_completed=[False,False]
        self.episode_reward_1=0
        self.episode_reward_2=0
        counts_1=np.zeros(7*9*7*9*5*2*5).reshape(7,9,7,9,5,2,5).astype(np.float32)
        counts_2=np.zeros(7*9*7*9*5*2*5).reshape(7,9,7,9,5,2,5).astype(np.float32)

        act0,act1=self.M_step_action_selection(self.env,self.env.get_state_single(list(map(int,state)),0),self.env.get_state_single(list(map(int,state)),1),pi_i_1,pi_1_2)
        next_state, next_rewards,move_completed=self.env.step([act0,act1]) 

        a1_state=self.env.get_state_single(list(map(int,state)),0)
        a2_state=self.env.get_state_single(list(map(int,state)),1)
        a1_next_state=self.env.get_state_single(list(map(int,next_state)),0)
        a2_next_state=self.env.get_state_single(list(map(int,next_state)),1)
        act0=int(act0)
        act1=int(act1)
        counts_1[tuple(a1_state+[act0])]+=1
        counts_2[tuple(a1_state+[act1])]+=1

        state=next_state
        rewards=next_rewards
        for t in range(1,self.turn_limit+1):
            if all(move_completed) and all(reward == 0 for reward in rewards):
                return counts_1,counts_2,self.episode_reward_1,self.episode_reward_2
            
                       
            self.episode_reward_1 += rewards[0]
            self.episode_reward_2 += rewards[1]


            act0,act1=self.M_step_action_selection(self.env,self.env.get_state_single(list(map(int,state)),0),self.env.get_state_single(list(map(int,state)),1),pi_i_1,pi_1_2)

            next_state, next_rewards,move_completed=self.env.step([act0,act1]) 

            assert len(next_state) == 8
            a1_state=self.env.get_state_single(list(map(int,state)),0)
            a2_state=self.env.get_state_single(list(map(int,state)),1)
            a1_next_state=self.env.get_state_single(list(map(int,next_state)),0)
            a2_next_state=self.env.get_state_single(list(map(int,next_state)),1)

        
            act0=int(act0)
            act1=int(act1)

            # counts discounted by discount rate as in formula
            counts_1[tuple(a1_state+[act0])]+=self.gamma**t
            counts_2[tuple(a1_state+[act1])]+=self.gamma**t
    
            state=next_state
            rewards=next_rewards

        
        #print(str(t+1)+" steps")        
        return counts_1,counts_2,self.episode_reward_1,self.episode_reward_2
        