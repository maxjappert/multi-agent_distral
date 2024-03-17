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

    def __init__(self, plan):
        self.actions = [NOOP, UP, DOWN, LEFT, RIGHT]
        self.inv_actions = [0, 2, 1, 4, 3]
        self.action_space = spaces.Discrete(5)
        self.action_pos_dict = {NOOP: [0, 0], UP: [-1, 0], DOWN: [1, 0], LEFT: [0, -1], RIGHT: [0, 1]}

        self.img_shape = [256, 256, 3]  # visualize state

        # initialize system state
        this_file_path = os.path.dirname(os.path.realpath(__file__))
        self.grid_map_path = os.path.join(this_file_path, 'plan{}.txt'.format(plan))
        self.start_grid_map = self._read_grid_map(self.grid_map_path)  # initial grid map
        self.current_grid_map = copy.deepcopy(self.start_grid_map)  # current grid map
        self.grid_map_shape = self.start_grid_map.shape
        self.observation_space = spaces.Box(low=np.array([-1.0, -1.0, -1.0]),
                                            high=np.array([1.0, 1.0, 1.0]))


        # OK so here it gets confusing: The term "state" is overdefined
        # The agent states are simply their coordinates, while the game state is a
        # sextuplet of (a1_normalised_coords, a1_action, a1_reward, a2_ ...)
        # I'm scared of refactoring this though, as I don't want to break anything

        # agent state: start, target, current state
        (self.agent1_start_state, self.agent1_target_state,
         self.agent2_start_state, self.agent2_target_state) = self._get_agents_start_target_state()

        self.agent_start_states = (self.agent1_start_state, self.agent2_start_state)
        self.agent_target_states = (self.agent1_target_state, self.agent2_target_state)
        self.agent_states = [copy.deepcopy(self.agent1_start_state), copy.deepcopy(self.agent2_start_state)]

        self.current_game_state = np.asarray([self.normalise_coordinates(self.agent_states[0]), 0., 0.,
                                              self.normalise_coordinates(self.agent_states[1]), 0., 0.])

        self.restart_once_done = False

        self.seed()

        self.episode_total_reward = 0.0

        # for gym
        self.viewer = None

    def normalise_coordinates(self, coords):
        return 2. * (self.grid_map_shape[0] * coords[0] + coords[1]) / (
                self.grid_map_shape[0] * self.grid_map_shape[1]) - 1.

    def seed(self, seed=None):

        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def get_state_single(self, coordinates, action, reward, agent_idx):
        """
        Gets an updated game state (sextuplet), where one opposing player to the one mentioned by agent_idx
        stays fix (their coordinates, action, reward), while the agent_idx player is updated.
        """
        op_game_state = self.current_game_state[:3] if agent_idx == 1 else self.current_game_state[3:6]

        you_new_game_state = np.asarray([self.normalise_coordinates(coordinates), (action - 2.5) / 5., reward])

        return np.concatenate([you_new_game_state, op_game_state]) if agent_idx == 0 else np.concatenate(
            [op_game_state, you_new_game_state])

    def step(self, action, agent_idx):

        # Return next observation, reward, finished, success

        action = int(action)
        info = {'success': False}
        done = False

        # Penalties
        penalty_step = 0.1
        penalty_wall = 0.5

        reward = -penalty_step
        nxt_agent_state = (self.agent_states[agent_idx][0] + self.action_pos_dict[action][0],
                           self.agent_states[agent_idx][1] + self.action_pos_dict[action][1])

        if action == NOOP:
            info['success'] = True
            self.episode_total_reward += reward  # Update total reward
            return self.get_state_single(self.agent_states[agent_idx], action, reward, agent_idx), reward, False, info

        op_idx = 0 if agent_idx == 1 else 1

        # Make a step

        # The move is illegal if it moves the agent out of bounds or if it collies with the opponent
        is_illegal_move = (nxt_agent_state[0] < 0 or nxt_agent_state[0] >= self.grid_map_shape[0]) or \
                          (nxt_agent_state[1] < 0 or nxt_agent_state[1] >= self.grid_map_shape[1]) or \
                          (math.isclose(nxt_agent_state[0], self.agent_states[op_idx][0]) and
                           math.isclose(nxt_agent_state[1], self.agent_states[op_idx][1]))

        if is_illegal_move:
            info['success'] = False
            self.episode_total_reward += reward  # Update total reward
            return self.get_state_single(self.agent_states[agent_idx], action, reward, agent_idx), reward, False, info

        target_position = self.current_grid_map[nxt_agent_state[0], nxt_agent_state[1]]

        if target_position == EMPTY:

            self.current_grid_map[nxt_agent_state[0], nxt_agent_state[1]] = AGENT1 if agent_idx == 0 else AGENT2

        elif target_position == WALL:

            info['success'] = False
            self.episode_total_reward += (reward - penalty_wall)  # Update total reward
            return self.get_state_single(self.agent_states[agent_idx], action, reward - penalty_wall, agent_idx), (
                    reward - penalty_wall), False, info

        elif target_position == TARGET1 if agent_idx == 0 else TARGET2:

            self.current_grid_map[nxt_agent_state[0], nxt_agent_state[1]] = SUCCESS

        self.current_grid_map[self.agent_states[agent_idx][0], self.agent_states[agent_idx][1]] = EMPTY
        self.agent_states[agent_idx] = copy.deepcopy(nxt_agent_state)
        info['success'] = True

        if nxt_agent_state[0] == self.agent_target_states[agent_idx][0] and nxt_agent_state[1] == \
                self.agent_target_states[agent_idx][1]:
            done = True
            reward += 1.0
            if self.restart_once_done:
                self.reset()

        self.episode_total_reward += reward  # Update total reward
        return self.get_state_single(self.agent_states[agent_idx], action, reward, agent_idx), reward, done, info

    def reset(self):

        # Return the initial two states of the environment

        self.agent_states = [copy.deepcopy(self.agent1_start_state), copy.deepcopy(self.agent2_start_state)]
        self.current_grid_map = copy.deepcopy(self.start_grid_map)
        self.episode_total_reward = 0.0

        # This is the normalising code copied from the authors adapted to two players
        return [self.normalise_coordinates(self.agent_states[0]), (0.0 - 2.5) / 5., 0.0,
                self.normalise_coordinates(self.agent_states[1]), (0.0 - 2.5) / 5., 0.0]

    def close(self):
        self.viewer.close() if self.viewer else None

    def _read_grid_map(self, grid_map_path):

        # Return the gridmap imported from a txt plan

        grid_map = open(grid_map_path, 'r').readlines()
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
