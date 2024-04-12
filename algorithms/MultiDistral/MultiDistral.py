import numpy as np
import math

from matplotlib import pyplot as plt
import scipy
from gridworld_env import GridworldEnv

class MultiDistral:
    def __init__(self, TURN_LIMIT, ALPHA, GAMMA,BETA,ALGO):

        self.episode_reward_1 = 0.0
        self.episode_reward_2 = 0.0
        self.pi_0_1 = np.ones(7*9*7*9*5*2*5).reshape(7,9,7,9,5,2,5).astype(np.float32) / (5)  # Initialize uniform prior policy
        self.pi_0_2 = np.ones(7*9*7*9*5*2*5).reshape(7,9,7,9,5,2,5).astype(np.float32) / (5)  # Initialize uniform prior policy
        self.turn_limit = TURN_LIMIT
        self.alpha = ALPHA
        self.gamma = GAMMA
        self.algorithm=ALGO
        self.eps = 1e-8
        self.turn_limit=TURN_LIMIT
        self.env1=GridworldEnv('1')
        self.env2=GridworldEnv('2')
        self.env3=GridworldEnv('3')
        # comment line below, and remove self.env4 from list in line 28 to test without tast 4
        self.env4=GridworldEnv('4')
        self.env5=GridworldEnv('5')

        self.list_envs=[self.env1, self.env2, self.env3, self.env4,self.env5]
        self.beta=BETA

    def optimise(self,version):

        iterations=10

        p1_total_rewards = np.zeros((iterations,))
        p2_total_rewards = np.zeros((iterations,))

        num_games = 30

        for i in range(iterations):

            print(f'Iteration {i+1}')
               
            # E STEP
            agents=[]
            p_player1=[]
            p_player2=[]

            # Alpha in the Distral paper! 
            PSI=0.7

            # beta in the paper
            BETA=self.beta


            # E Step

            if version==1:
                for j,env in enumerate(self.list_envs):
                    env.reset()
                    agents.append(self.algorithm(self.list_envs[j],num_games,self.turn_limit,self.alpha,self.gamma,PSI,BETA,version,self.pi_0_1,self.pi_0_2))

                    for game in range(num_games):
                        _, reward1, reward2 = agents[j].learn()
                        p1_total_rewards[i] += reward1
                        p2_total_rewards[i] += reward2

                    # Extract pi_i from E step (for comp reasons, this isnt a proper extraction, which would involve applying action_selection() to each state)
                    # we simply use the latest pi_i for each state. We have 2 pi_i (one for each player) for each task (in total 10 pi_i)
                    p_player1.append(agents[j].pi1)
                    p_player2.append(agents[j].pi2)

            elif version==2:
                for j,env in enumerate(self.list_envs):
                    env.reset()
                    agents.append(self.algorithm(self.list_envs[j],num_games,self.turn_limit,self.alpha,self.gamma,PSI,BETA,version,self.pi_0_1,self.pi_0_2))

                    for game in range(num_games):
                        _, reward1, reward2 = agents[j].learn()
                        p1_total_rewards[i] += reward1
                        p2_total_rewards[i] += reward2

                    # Extract pi_i from E step (for comp reasons, this isnt a proper extraction, which would involve applying action_selection() to each state)
                    # we simply use the latest pi_i for each state. We have 2 pi_i (one for each player) for each task (in total 10 pi_i)
                    p_player1.append(agents[j].pi1)
                    p_player2.append(agents[j].pi2)

                

            # M STEP

            if version==1: # 1 distilled policy per agent 
                total_counts_1=np.zeros(7*9*7*9*5*2*5).reshape(7,9,7,9,5,2,5).astype(np.float32)
                total_counts_2=np.zeros(7*9*7*9*5*2*5).reshape(7,9,7,9,5,2,5).astype(np.float32)
                for j,env in enumerate(self.list_envs):
                    self.list_envs[j].reset()
                    agents.append(self.algorithm(self.list_envs[j],num_games,self.turn_limit,self.alpha,self.gamma,PSI,BETA,version,self.pi_0_1,self.pi_0_2))
                    for game in range(num_games):
                        counts1,counts2,_,_=agents[j].M_step(p_player1[j],p_player2[j])
                        total_counts_1+=counts1
                        total_counts_2+=counts2
                
                sum_over_actions_1 = np.sum(np.maximum(total_counts_1, self.eps), axis=-1, keepdims=True)
                sum_over_actions_2 = np.sum(np.maximum(total_counts_2, self.eps), axis=-1, keepdims=True)

                new_pi_0_1 = np.maximum(total_counts_1, self.eps) / sum_over_actions_1
                new_pi_0_2 = np.maximum(total_counts_2, self.eps) / sum_over_actions_2

                self.pi_0_1=new_pi_0_1
                self.pi_0_2=new_pi_0_2

                assert np.all(new_pi_0_1 >= 0) and np.all(new_pi_0_1 <= 1), "Probabilities must be between 0 and 1."
                assert np.allclose(np.sum(new_pi_0_1, axis=-1), 1), "Sum of probabilities for each state must equal 1."
                assert np.all(new_pi_0_2 >= 0) and np.all(new_pi_0_2 <= 1), "Probabilities must be between 0 and 1."
                assert np.allclose(np.sum(new_pi_0_2, axis=-1), 1), "Sum of probabilities for each state must equal 1."
            elif version == 2:  # 1 shared policy across all AGENTS
                total_counts = np.zeros(7 * 9 * 7 * 9 * 5 * 2 * 5).reshape(7, 9, 7, 9, 5, 2, 5).astype(np.float32)
                for j, env in enumerate(self.list_envs):
                    self.list_envs[j].reset()
                    agents.append(
                        self.algorithm(self.list_envs[j], num_games, self.turn_limit, self.alpha, self.gamma,
                                       PSI, BETA, version, self.pi_0_1, self.pi_0_2))
                    for game in range(num_games):
                        counts1, counts2, _, _ = agents[j].M_step(p_player1[j], p_player2[j])
                        total_counts += counts1
                        total_counts += counts2

                # Add a small positive constant to counts to avoid zero probabilities (Laplace smoothing)
                pseudo_count = 1e-8
                adjusted_counts = total_counts + pseudo_count

                # Normalize the adjusted counts to obtain a valid probability distribution
                sum_over_actions = np.sum(adjusted_counts, axis=-1, keepdims=True)
                new_pi_0 = adjusted_counts / np.maximum(sum_over_actions, self.eps)

                # Change pi_0 just acquired
                self.pi_0_1 = new_pi_0
                self.pi_0_2 = new_pi_0

                # Assertions to verify that new_pi_0 forms a valid probability distribution
                assert np.all(new_pi_0 >= 0) and np.all(new_pi_0 <= 1), "Probabilities must be between 0 and 1."
                assert np.allclose(np.sum(new_pi_0, axis=-1), 1), "Sum of probabilities for each state must equal 1."

        plt.plot(list(range(1,iterations+1)), p1_total_rewards/(num_games*len(self.list_envs)))
        plt.plot(list(range(1,iterations+1)), p2_total_rewards/(num_games*len(self.list_envs)))
