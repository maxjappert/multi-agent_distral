import math

import sys
import os
import copy
from gym import spaces
import numpy as np
import matplotlib.pyplot as plt
from gym.utils import seeding
from numba import jit

# Used for the plan.txt files to define an environment
EMPTY = BLACK = 0
WALL = GRAY = 1
TARGET1 = D_GREEN = 2
TARGET2 = D_RED = 3
AGENT1 = L_GREEN = 4
AGENT2 = L_RED = 5
SUCCESS = PINK = 6

# NOTe to Max: This isn't necessarily something you need to do, I'm just thinking abt whether this env makes it easy to connect to algorithms.
# So the goal would be to do tabular RL: we would basically have a very large data structure with all the variables that define an agent's state
# (i.e. the sextuple defined below) and then use an algo to learn the value of each of these (think of it like Q-Learning).
# Just wanted you to think whether the environment makes sense to do this, since you know the code better.

# NOTe to MAX: (the colour chocie is confusing imo, better if agent and its goal are same colour (but e.g. diff shades))
# I made it so that agent 1 is a dark and agent 2 a light shade of green/red
# This is only for visualising the environment
COLORS = {BLACK: [0.0, 0.0, 0.0], GRAY: [0.5, 0.5, 0.5], D_GREEN: [0.0, 0.4, 0.0],
          D_RED: [0.4, 0.0, 0.0], L_GREEN: [0.5, 1.0, 0.5], L_RED: [1.0, 0.5, 0.5], PINK: [1.0, 0.0, 1.0]}

NOOP = 0
DOWN = 1
UP = 2
LEFT = 3
RIGHT = 4


class GridworldEnv:
    metadata = {'render.modes': ['human', 'rgb_array']}
    num_env = 0

    def __init__(self, plan, plan_txt=False):
        # NOOP means not moving
        self.actions = [NOOP, UP, DOWN, LEFT, RIGHT]

        # No idea what this is, not used anywhere in the code
        self.inv_actions = [0, 2, 1, 4, 3]

        # Also not used anywhere
        self.action_space = {0:spaces.Discrete(5),1:spaces.Discrete(5)}
        #self.action_space_p1 = spaces.Discrete(5)

        # Position updates for each action
        self.action_pos_dict = {NOOP: [0, 0], UP: [-1, 0], DOWN: [1, 0], LEFT: [0, -1], RIGHT: [0, 1]}

        # For state visualisation
        self.img_shape = [256, 256, 3]

        # Initialise the state from either a plan.txt file or directly from string
        this_file_path = os.path.dirname(os.path.realpath(__file__))
        if not plan_txt:
            self.grid_map_path = os.path.join(this_file_path, 'plan{}.txt'.format(plan))
            self.start_grid_map = self._read_grid_map(self.grid_map_path)  # initial grid map
        else:
            self.start_grid_map = plan

        # The grid map is a 2D-array of integers representing what each square is (wall, goal, agent, etc.)
        self.current_grid_map = copy.deepcopy(self.start_grid_map)  # current grid map
        self.grid_map_shape = self.start_grid_map.shape

        # Specify bounds of observation space (we consider fully observable environment)
        # Changed THIS, ISNT OBSERVATION A 5-TUPLE NOW?
        self.observation_space = spaces.Box(low=np.array([-1.0, -1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0]),
                                            high=np.array([8.0, 9.0,8.0,9.0, 5.0,1.0,5.0,1.0]))

        # agent state: start, target, current state
        (self.agent1_start_coords, self.agent1_target_coords,
         self.agent2_start_coords, self.agent2_target_coords) = self._get_agents_start_target_state()

        self.agents_start_coords = (self.agent1_start_coords, self.agent2_start_coords)
        self.agents_target_coords = (self.agent1_target_coords, self.agent2_target_coords)
        self.current_agents_coords = [copy.deepcopy(self.agent1_start_coords), copy.deepcopy(self.agent2_start_coords)]

        # Game state: (p1 y coord, p1 x coord, p1action, p1reward, p2 y coord, p2 x coord, p2action, p2reward)
        self.current_game_state = np.asarray([self.agents_start_coords[0][0],self.agents_start_coords[0][1], self.agents_start_coords[1][0],self.agents_start_coords[1][1],
                                              0., 0.,
                                               0., 0.])

        self.restart_once_done = False

        # Set seed
        self.seed()

        self.episode_total_reward = 0.0
        self.move_completed=[False,False]
        self.p1_total_reward=0
        self.p2_total_reward=0

        # for gym
        self.viewer = None

    # NOte to Max: they say in function get_state that this is for better performance of NN. So perhaps could u pls remove this normalisation?
    # we dont need it in tabular

    #def normalise_coordinates(self, coords):
    #    return 2. * (self.grid_map_shape[0] * coords[0] + coords[1]) / (
    #            self.grid_map_shape[0] * self.grid_map_shape[1]) - 1.

    def seed(self, seed=None):

        self.np_random, seed = seeding.np_random(seed)
        return [seed]


    def get_state_single(self, state, agent_idx):
        """
        Returns the state for the specified agent from the combined state.
        """
        # state of env has coordinates p1 +coordinates p2 + p1 prev action +  p1 reward so far + p2 prev action  + p2 reward so far
        # state of each agent should be coordinates p1 + coordinates p2 + their own prev action  + other agent's reward so far
        if agent_idx==0:
            return state[:5]+[state[7]]
        else:
            return state[:4]+[state[6]]+[state[5]]
        
    # Changed this so both agents act at the same time

    def step(self, actions):
        """
        Performs one step for both agents, given their actions
        Returns: new_state, reward, move_completed, info
        """

        # Return next observation, reward, finished, success
        #wall_penalty=0.5
        rewards = [0.0, 0.0]
        #these will be actual coordinates after step
        new_agent_coords=[]
        #these will be tentative coordinates while we check if step is valid
        updated_agent_coords_list=[]
        for agent_idx,action in enumerate(actions):
            action=int(action)

            if agent_idx==0 and self.move_completed[0]:
                action=0
            elif agent_idx==1 and self.move_completed[1]:
                action=0

            # Coordinates of each agent based on their action
            # note these might not be their final coordinates since we need to check validity of step
            updated_agent_coords = (self.current_agents_coords[agent_idx][0] + self.action_pos_dict[action][0],
                                self.current_agents_coords[agent_idx][1] + self.action_pos_dict[action][1])
            updated_agent_coords_list.append(updated_agent_coords)
        
        # Check if the move is illegal
        for agent_idx,action in enumerate(actions):
            
            if self.move_completed[agent_idx]:

                rewards[agent_idx]=0.0
                new_agent_coords.append(self.current_agents_coords[agent_idx])
                updated_agent_coords_list[agent_idx]=self.current_agents_coords[agent_idx]

                continue
            # Agent doesn't have to be moved if action is NOOP
            if action == NOOP:
                
                rewards[agent_idx]=0.0
                new_agent_coords.append(self.current_agents_coords[agent_idx])
                updated_agent_coords_list[agent_idx]=self.current_agents_coords[agent_idx]
                continue
                #new_agent_coords.append(self.current_agents_coords[agent_idx])
                #continue #go to next player

            if agent_idx==0:
                opponent=1
            else:
                opponent=0
            # The move is illegal if it moves the agent out of bounds or if it collies with the opponent
            is_illegal_move = (updated_agent_coords_list[agent_idx][0] < 0 or updated_agent_coords_list[agent_idx][0] >= self.grid_map_shape[0]) or \
                            (updated_agent_coords_list[agent_idx][1] < 0 or updated_agent_coords_list[agent_idx][1] >= self.grid_map_shape[1]) or \
                            (updated_agent_coords_list[agent_idx]==updated_agent_coords_list[opponent]) or \
                            (updated_agent_coords_list[agent_idx]==self.current_agents_coords[opponent])

            #if it's illegal, both agents stay at the same spot
            if is_illegal_move:
                
                rewards[agent_idx] = 0.0  # Update reward
                new_agent_coords.append(self.current_agents_coords[agent_idx])
                updated_agent_coords_list[agent_idx]=self.current_agents_coords[agent_idx]
                continue

            target_position = self.current_grid_map[updated_agent_coords_list[agent_idx][0], updated_agent_coords_list[agent_idx][1]]

            # Update the grid map
            if target_position == EMPTY:
                if agent_idx==0:
                    self.current_grid_map[updated_agent_coords_list[agent_idx][0], updated_agent_coords_list[agent_idx][1]] = AGENT1
                elif agent_idx==1:
                    self.current_grid_map[updated_agent_coords_list[agent_idx][0], updated_agent_coords_list[agent_idx][1]] = AGENT2
            elif target_position == WALL:
                
                rewards[agent_idx] = -1
                new_agent_coords.append(self.current_agents_coords[agent_idx])
                updated_agent_coords_list[agent_idx]=self.current_agents_coords[agent_idx]
                continue
            elif agent_idx==0 and target_position == TARGET1:

                self.current_grid_map[updated_agent_coords_list[agent_idx][0], updated_agent_coords_list[agent_idx][1]] = SUCCESS
                self.move_completed[agent_idx] = True
                rewards[agent_idx] = 1.0
                self.p1_total_reward+=1
            elif agent_idx==1 and target_position==TARGET2:

                self.current_grid_map[updated_agent_coords_list[agent_idx][0], updated_agent_coords_list[agent_idx][1]] = SUCCESS
                self.move_completed[agent_idx] = True
                rewards[agent_idx] = 1.0
                self.p2_total_reward+=1
        
            # Replace the old agent coordinates with value of previous state (might be blank, or the opponent's goal, or where the opponent moves to)
            if updated_agent_coords_list[agent_idx]!=self.current_agents_coords[agent_idx]:

                # if state is a target state
                if copy.deepcopy(self.start_grid_map[self.current_agents_coords[agent_idx][0], self.current_agents_coords[agent_idx][1]])==TARGET2:
                    self.current_grid_map[self.current_agents_coords[agent_idx][0], self.current_agents_coords[agent_idx][1]] = TARGET2
                elif copy.deepcopy(self.start_grid_map[self.current_agents_coords[agent_idx][0], self.current_agents_coords[agent_idx][1]])==TARGET1:
                    self.current_grid_map[self.current_agents_coords[agent_idx][0], self.current_agents_coords[agent_idx][1]] = TARGET1

                # if the opponent moves to the square the player was just in. (this is no longer allowed!)
                # Note if the current player is player 1 (idx=0), we dont want to update the square to empty, as that would mess with the checks of valid move for player 2
                #elif updated_agent_coords_list[opponent]== self.current_agents_coords[agent_idx]:
                #    if opponent==0:
                 #       self.current_grid_map[self.current_agents_coords[agent_idx][0], self.current_agents_coords[agent_idx][1]] = AGENT1
                else:
                    self.current_grid_map[self.current_agents_coords[agent_idx][0], self.current_agents_coords[agent_idx][1]] = EMPTY

            # Apply update
            new_agent_coords.append(updated_agent_coords_list[agent_idx])

            self.episode_total_reward += rewards[agent_idx]  # Update total reward

            if False not in self.move_completed:

                new_state = np.asarray([updated_agent_coords_list[0][0], updated_agent_coords_list[0][1], updated_agent_coords_list[1][0], 
                                        updated_agent_coords_list[1][1], actions[0], rewards[0], actions[1], rewards[1]])
                return new_state, rewards, self.move_completed
            
        self.current_agents_coords=new_agent_coords
        new_state = np.asarray([new_agent_coords[0][0], new_agent_coords[0][1], new_agent_coords[1][0], new_agent_coords[1][1], 
                                actions[0], self.p1_total_reward,
                                 actions[1], self.p2_total_reward])

    
        return new_state, rewards, self.move_completed


    def reset(self):

        # Return the initial two states of the environment

        self.current_agents_coords = [copy.deepcopy(self.agent1_start_coords), copy.deepcopy(self.agent2_start_coords)]
        self.current_grid_map = copy.deepcopy(self.start_grid_map)
        self.episode_total_reward = 0.0
        self.move_completed=[False,False]
        self.p1_total_reward=0
        self.p2_total_reward=0
        # This is the normalising code copied from the authors adapted to two players
        return [self.current_agents_coords[0][0],self.current_agents_coords[0][1], self.current_agents_coords[1][0],self.current_agents_coords[1][1], 0.0, 0.0,
                 0.0, 0.0]

    def close(self):
        self.viewer.close() if self.viewer else None

    def _read_grid_map(self, grid_map_path, file=True):

        # Return the gridmap imported from a txt plan

        if file:
            grid_map = open(grid_map_path, 'r').readlines()
        else:
            grid_map = grid_map_path

        grid_map_array = []
        for k1 in grid_map:
            k1s = k1.split(' ')
            tmp_arr = []
            for k2 in k1s:
                try:
                    tmp_arr.append(int(k2))
                except:
                    pass
            grid_map_array.append(tmp_arr)
        grid_map_array = np.array(grid_map_array, dtype=int)
        return grid_map_array

    def _get_agents_start_target_state(self):
        start_state_1 = np.where(self.start_grid_map == AGENT1)
        start_state_2 = np.where(self.start_grid_map == AGENT2)
        target_state_1 = np.where(self.start_grid_map == TARGET1)
        target_state_2 = np.where(self.start_grid_map == TARGET2)

        # if neither state is found
        start_or_target_not_found = not (start_state_1[0] and target_state_1[0]) or not (
                start_state_2[0] and target_state_2[0])
        if start_or_target_not_found:
            sys.exit('Start or target state not specified')
        start_state_1 = (start_state_1[0][0], start_state_1[1][0])
        start_state_2 = (start_state_2[0][0], start_state_2[1][0])
        target_state_1 = (target_state_1[0][0], target_state_1[1][0])
        target_state_2 = (target_state_2[0][0], target_state_2[1][0])

        return start_state_1, target_state_1, start_state_2, target_state_2

    def _gridmap_to_image(self, img_shape=None):

        # Return image from the gridmap

        if img_shape is None:
            img_shape = self.img_shape
        observation = np.random.randn(*img_shape) * 0.0
        gs0 = int(observation.shape[0] / self.current_grid_map.shape[0])
        gs1 = int(observation.shape[1] / self.current_grid_map.shape[1])
        for i in range(self.current_grid_map.shape[0]):
            for j in range(self.current_grid_map.shape[1]):
                for k in range(3):
                    this_value = COLORS[self.current_grid_map[i, j]][k]
                    observation[i * gs0:(i + 1) * gs0, j * gs1:(j + 1) * gs1, k] = this_value
        return (255 * observation).astype(np.uint8)

    def render(self, mode='human', close=False,aspect='auto'):

        # Returns a visualization of the environment according to specification

        if close:
            plt.close(1)  # Final plot
            return

        img = self._gridmap_to_image()
        if mode == 'rgb_array':
            return img
        elif mode == 'human':
            plt.figure()
            plt.imshow(img)
            return