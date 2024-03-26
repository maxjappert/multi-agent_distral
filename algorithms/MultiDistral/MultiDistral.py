import numpy as np
import math
from gridworld_env import GridworldEnv
from algorithms.MultiDistral.MultiDistral_E_step.Soft_Q_Learning_without_rollout import Soft_without_rollout

class MultiDistral:
    def __init__(self, TURN_LIMIT, ALPHA, GAMMA, TAU,ALGO):
        #self.env = env
        self.episode_reward_1 = 0.0
        self.episode_reward_2 = 0.0
        #self.pi_0 = np.ones(7*9*7*9*5*2*5).reshape(7,9,7,9,5,2,5).astype(np.float32) / (5)  # Initialize uniform prior policy
        self.pi_0_1 = np.ones(7*9*7*9*5*2*5).reshape(7,9,7,9,5,2,5).astype(np.float32) / (5)  # Initialize uniform prior policy
        self.pi_0_2 = np.ones(7*9*7*9*5*2*5).reshape(7,9,7,9,5,2,5).astype(np.float32) / (5)  # Initialize uniform prior policy
        self.turn_limit = TURN_LIMIT
        self.alpha = ALPHA
        self.gamma = GAMMA
        self.tau = TAU
        self.algorithm=ALGO
        self.turn_limit=TURN_LIMIT
        self.env1=GridworldEnv('1')
        self.env2=GridworldEnv('2')
        self.env3=GridworldEnv('3')
        self.env4=GridworldEnv('4')
        self.env5=GridworldEnv('5')
        self.list_envs=[self.env1,self.env2,self.env3,self.env4,self.env5]

    def optimise(self,version):

        iterations=2
        for i in range(iterations):
               
            # E STEP
            num_games=10
            agents=[]
            p_player1=[]
            p_player2=[]
            PSI=0.7
            BETA=0.5

            if version==1:
                for i,env in enumerate(self.list_envs):
                    env.reset()
                    agents.append(self.algorithm(self.list_envs[i],num_games,self.turn_limit,self.alpha,self.gamma,self.tau,PSI,BETA,version,self.pi_0_1,self.pi_0_2))
                    for game in range(num_games):
                        agents[i].learn()

                    # Extract pi_i from E step (for comp reasons, this isnt a proper extraction, which would involve applying action_selection() to each state)
                    # we simply use the latest pi_i for each state. We have 2 pi_i (one for each player) for each task (in total 10 pi_i)
                    p_player1.append(agents[i].pi1)
                    p_player2.append(agents[i].pi2)

            elif version==2:
                for i,env in enumerate(self.list_envs):
                    env.reset()
                    agents.append(self.algorithm(self.list_envs[i],num_games,self.turn_limit,self.alpha,self.gamma,self.tau,PSI,BETA,version,self.pi_0_1,self.pi_0_2))
                    for game in range(num_games):
                        agents[i].learn()

                    # Extract pi_i from E step (for comp reasons, this isnt a proper extraction, which would involve applying action_selection() to each state)
                    # we simply use the latest pi_i for each state. We have 2 pi_i (one for each player) for each task (in total 10 pi_i)
                    p_player1.append(agents[i].pi1)
                    p_player2.append(agents[i].pi2)

                

            # M STEP
                
            if version==1: # 1 distilled policy per agent 
                total_counts_1=np.zeros(7*9*7*9*5*2*5).reshape(7,9,7,9,5,2,5).astype(np.float32)
                total_counts_2=np.zeros(7*9*7*9*5*2*5).reshape(7,9,7,9,5,2,5).astype(np.float32)
                for i,env in enumerate(self.list_envs):
                    self.list_envs[i].reset()
                    agents.append(self.algorithm(self.list_envs[i],num_games,self.turn_limit,self.alpha,self.gamma,self.tau,PSI,BETA,version,self.pi_0_1,self.pi_0_2))
                    for game in range(num_games):
                        counts1,counts2,_,_=agents[i].M_step(p_player1[i],p_player2[i])
                        total_counts_1+=counts1
                        total_counts_2+=counts2
                
                new_pi_0_1=total_counts_1/np.apply_over_axes(np.sum, total_counts_1, range(total_counts_1.ndim - 1)) #sum over all but last axis
                new_pi_0_2=total_counts_2/np.apply_over_axes(np.sum, total_counts_2, range(total_counts_2.ndim - 1)) #sum over all but last axis
                self.pi_0_1=new_pi_0_1 
                self.pi_0_2=new_pi_0_2
                
            elif version==2: #1 shared policy across all AGENTS
                total_counts=np.zeros(7*9*7*9*5*2*5).reshape(7,9,7,9,5,2,5).astype(np.float32)
                for i,env in enumerate(self.list_envs):
                    self.list_envs[i].reset()
                    agents.append(self.algorithm(self.list_envs[i],num_games,self.turn_limit,self.alpha,self.gamma,self.tau,PSI,BETA,version,self.pi_0_1,self.pi_0_2))
                    for game in range(num_games):
                        counts1,counts2,_,_=agents[i].M_step(p_player1[i],p_player2[i])
                        total_counts+=counts1
                        total_counts+=counts2
                
                new_pi_0=total_counts/np.apply_over_axes(np.sum, total_counts, range(total_counts.ndim - 1)) #sum over all but last axis

                #print(new_pi_0[1,1,1,7,0,0,:])
                # Change pi_0 just acquired
                self.pi_0_1=new_pi_0
                self.pi_0_2=new_pi_0