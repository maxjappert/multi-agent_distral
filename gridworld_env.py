import math

import sys
import os
import copy
from gym import spaces
import numpy as np
import matplotlib.pyplot as plt
from gym.utils import seeding

# Used for the plan.txt files to define an environment
EMPTY = BLACK = 0
WALL = GRAY = 1
TARGET1 = D_GREEN = 2
TARGET2 = L_GREEN = 3
AGENT1 = D_RED = 4
AGENT2 = L_RED = 5
SUCCESS = PINK = 6

# NOTe to Max: This isn't necessarily something you need to do, I'm just thinking abt whether this env makes it easy to connect to algorithms.
# So the goal would be to do tabular RL: we would basically have a very large data structure with all the variables that define an agent's state
# (i.e. the sextuple defined below) and then use an algo to learn the value of each of these (think of it like Q-Learning).
# Just wanted you to think whether the environment makes sense to do this, since you know the code better.

# NOTe to MAX: (the colour chocie is confusing imo, better if agent and its goal are same colour (but e.g. diff shades))
# I made it so that agent 1 is a dark and agent 2 a light shade of green/red
# This is only for visualising the environment
COLORS = {BLACK: [0.0, 0.0, 0.0], GRAY: [0.5, 0.5, 0.5], D_GREEN: [0.0, 0.4, 0.0], L_GREEN: [0.5, 1.0, 0.5],
          D_RED: [0.4, 0.0, 0.0], L_RED: [1.0, 0.5, 0.5], PINK: [1.0, 0.0, 1.0]}

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
        self.action_space = spaces.Discrete(5)

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

        # Don't know what this is
        self.observation_space = spaces.Box(low=np.array([-1.0, -1.0, -1.0]),
                                            high=np.array([1.0, 1.0, 1.0]))

        # agent state: start, target, current state
        (self.agent1_start_coords, self.agent1_target_coords,
         self.agent2_start_coords, self.agent2_target_coords) = self._get_agents_start_target_state()

        self.agents_start_coords = (self.agent1_start_coords, self.agent2_start_coords)
        self.agents_target_coords = (self.agent1_target_coords, self.agent2_target_coords)
        self.current_agents_coords = [copy.deepcopy(self.agent1_start_coords), copy.deepcopy(self.agent2_start_coords)]

        # Game state: (p1coords, p1action, p1reward, p2coords, p2action, p2reward)
        self.current_game_state = np.asarray([self.reshape_coordinates(self.current_agents_coords[0]), 0., 0.,
                                              self.reshape_coordinates(self.current_agents_coords[1]), 0., 0.])

        self.restart_once_done = False

        # Set seed
        self.seed()

        self.episode_total_reward = 0.0

        # for gym
        self.viewer = None


    def reshape_coordinates(self, coords):
        """
        Reshapes coordinates to work in a 1D array. Formerly called normalise_coordinates.
        """
        return 2. * (self.grid_map_shape[0] * coords[0] + coords[1]) / (
                self.grid_map_shape[0] * self.grid_map_shape[1]) - 1.

    def seed(self, seed=None):

        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    # NOte: Max, is your code selecting the previous actions the agents took? 
    # See section 4.1. of paper: their state representation (for single agent) has current location, the previous action of the agent,
    # and the previous reward. Ideally, i think our state representation (for each agent, in Multi-agent setting!) 
    # would be 
    # the locations of both agents, the previous actions of both agents (just the latest, not entire history) and their previous reward!
    # hard for me to see if that's what u are doing in this function
    def get_state_single(self, coordinates, action, reward, agent_idx):
        """
        Gets an updated game state (sextuplet), where one opposing player to the one mentioned by agent_idx
        stays fix (their coordinates, action, reward), while the agent_idx player is updated.

        Game state:

        (p1coords, p1action, p1reward, p2coords, p2action, p2reward)
        """
        op_game_state = self.current_game_state[:3] if agent_idx == 1 else self.current_game_state[3:6]

        you_new_game_state = np.asarray([self.reshape_coordinates(coordinates), action, reward])

        return np.concatenate([you_new_game_state, op_game_state]) if agent_idx == 0 else np.concatenate(
            [op_game_state, you_new_game_state])

    # NOTe: This function is currently doing a step only for one of the agents. We need it to do the step for both!
    # Basically the agents need to take an action at the same time. If we separate it, player 2 knows where player 1 is moving
    def step(self, action, agent_idx):
        """
        Performs one step for the given agent and the given action.

        Returns: new_state, reward, move_completed, info
        """

        # Return next observation, reward, finished, success

        action = int(action)
        info = {'success': False}

        # Formerly named "done"
        move_completed = False

        # Penalties
        penalty_step = 0.1
        penalty_wall = 0.5

        # Each step is penalised in order to minimise the number of steps
        reward = -penalty_step
        updated_agent_coords = (self.current_agents_coords[agent_idx][0] + self.action_pos_dict[action][0],
                                self.current_agents_coords[agent_idx][1] + self.action_pos_dict[action][1])

        # Agent doesn't have to be moved if action is NOOP
        if action == NOOP:
            info['success'] = True
            self.episode_total_reward += reward  # Update total reward
            return self.get_state_single(self.current_agents_coords[agent_idx], action, reward, agent_idx), reward, False, info

        # The opponent index is just the other index to the player index
        op_idx = 0 if agent_idx == 1 else 1

        # Make a step

        # The move is illegal if it moves the agent out of bounds or if it collies with the opponent
        is_illegal_move = (updated_agent_coords[0] < 0 or updated_agent_coords[0] >= self.grid_map_shape[0]) or \
                          (updated_agent_coords[1] < 0 or updated_agent_coords[1] >= self.grid_map_shape[1]) or \
                          (math.isclose(updated_agent_coords[0], self.current_agents_coords[op_idx][0]) and
                           math.isclose(updated_agent_coords[1], self.current_agents_coords[op_idx][1]))

        if is_illegal_move:
            info['success'] = False
            self.episode_total_reward += reward  # Update total reward
            return self.get_state_single(self.current_agents_coords[agent_idx], action, reward, agent_idx), reward, False, info

        target_position = self.current_grid_map[updated_agent_coords[0], updated_agent_coords[1]]

        # Update the grid map

        if target_position == EMPTY:
            self.current_grid_map[updated_agent_coords[0], updated_agent_coords[1]] = AGENT1 if agent_idx == 0 else AGENT2
        # No idea why they check for this again
        elif target_position == WALL:
            info['success'] = False
            self.episode_total_reward += (reward - penalty_wall)  # Update total reward
            return self.get_state_single(self.current_agents_coords[agent_idx], action, reward - penalty_wall, agent_idx), (
                    reward - penalty_wall), False, info
        # If the agent has reached their target
        elif target_position == TARGET1 if agent_idx == 0 else TARGET2:
            self.current_grid_map[updated_agent_coords[0], updated_agent_coords[1]] = SUCCESS

        # Replace the old agent coordinates with empty space
        self.current_grid_map[self.current_agents_coords[agent_idx][0], self.current_agents_coords[agent_idx][1]] = EMPTY

        # Apply update
        self.current_agents_coords[agent_idx] = copy.deepcopy(updated_agent_coords)
        info['success'] = True

        # If agent has reached
        if updated_agent_coords[0] == self.agents_target_coords[agent_idx][0] and updated_agent_coords[1] == \
                self.agents_target_coords[agent_idx][1]:
            move_completed = True
            reward += 1.0
            if self.restart_once_done:
                self.reset()

        self.episode_total_reward += reward  # Update total reward
        return self.get_state_single(self.current_agents_coords[agent_idx], action, reward, agent_idx), reward, move_completed, info

    def reset(self):

        # Return the initial two states of the environment

        self.current_agents_coords = [copy.deepcopy(self.agent1_start_coords), copy.deepcopy(self.agent2_start_coords)]
        self.current_grid_map = copy.deepcopy(self.start_grid_map)
        self.episode_total_reward = 0.0

        # This is the normalising code copied from the authors adapted to two players
        return [self.reshape_coordinates(self.current_agents_coords[0]), 0.0, 0.0,
                self.reshape_coordinates(self.current_agents_coords[1]), 0.0, 0.0]

    def close(self):
        self.viewer.close() if self.viewer else None

    def _read_grid_map(self, grid_map_path, file=True):

        # Return the gridmap imported from a txt plan

        if file:
            grid_map = open(grid_map_path, 'r').readlines()
        else:
            grid_map = grid_map_path

        print(grid_map)
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

    def render(self, mode='human', close=False):

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
