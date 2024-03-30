import copy
import itertools
import os
import random
import sys

import numpy as np
from gym import spaces, Env
from gym.utils import seeding
import matplotlib.pyplot as plt

# Define the environment's elements
EMPTY, WALL, TARGET1, TARGET2, AGENT1, AGENT2, SUCCESS, TARGET1_OC2, TARGET2_OC1 = range(9)
ACTIONS = [NOOP, UP, DOWN, LEFT, RIGHT] = range(5)

# Define colors for rendering
COLORS = {
    EMPTY: [1.0, 1.0, 1.0],  # White
    WALL: [0.0, 0.0, 0.0],  # Black
    TARGET1: [0.0, 0.5, 0.0],  # Dark Green
    TARGET2: [0.5, 0.0, 0.0],  # Dark Red
    AGENT1: [0.5, 1.0, 0.5],  # Light Green
    AGENT2: [1.0, 0.5, 0.5],  # Light Red
    SUCCESS: [1.0, 0.0, 1.0],  # Pink
    TARGET1_OC2: [0.5, 0.5, 0.5],
    TARGET2_OC1: [0.5, 0.5, 0.5],
    TARGET2_OC1: [0.5, 0.5, 0.5]
}

# Define the action to position update mapping
ACTION_EFFECTS = {
    NOOP: (0, 0),
    UP: (-1, 0),
    DOWN: (1, 0),
    LEFT: (0, -1),
    RIGHT: (0, 1),
}


class GridworldEnv(Env):
    metadata = {'render.modes': ['human', 'rgb_array']}

    def __init__(self, plan, from_file=True):
        super(GridworldEnv, self).__init__()
        self.action_space = spaces.Discrete(len(ACTIONS))
        self.action_combinations = list(itertools.product(ACTIONS, repeat=2))

        # Load the grid map
        if from_file:
            this_file_path = os.path.dirname(os.path.realpath(__file__))
            grid_map_path = os.path.join(this_file_path, 'tasks/task{}.txt'.format(plan))
            self.start_grid_map = self._read_grid_map(grid_map_path)
        else:
            self.start_grid_map = np.array(plan, dtype=int)

        self.current_grid_map = np.copy(self.start_grid_map)
        self.observation_space = spaces.Box(low=0, high=max(EMPTY, WALL, TARGET1, TARGET2, AGENT1, AGENT2, SUCCESS),
                                            shape=self.current_grid_map.shape, dtype=int)

        # Initialize agents' states
        self.agents_start_coords, self.agents_target_coords = self._find_agents_and_targets()
        self.current_agents_coords = np.copy(self.agents_start_coords)
        self.move_completed = [False, False]
        self.episode_total_reward = 0.0
        self.viewer = None
        self.seed()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def get_legal_action_pairs(self):
        legal_action_pairs = []

        for comb in self.action_combinations:
            if self.move_legal(comb):
                legal_action_pairs.append(comb)

        return legal_action_pairs

    def move_legal(self, action_pair):

        # If an agent has reached their goal, they can only remain where they are
        if self.move_completed[0] == True and action_pair[0] != NOOP:
            return False

        if self.move_completed[1] == True and action_pair[1] != NOOP:
            return False

        dy_1, dx_1 = ACTION_EFFECTS[action_pair[0]]
        dy_2, dx_2 = ACTION_EFFECTS[action_pair[1]]
        y_1, x_1 = self.current_agents_coords[0][0] + dy_1, self.current_agents_coords[0][1] + dx_1
        y_2, x_2 = self.current_agents_coords[1][0] + dy_2, self.current_agents_coords[1][1] + dx_2

        return (self._within_bounds(y_1, x_1) and self._within_bounds(y_2, x_2)
                and self.current_grid_map[y_1, x_1] != WALL and self.current_grid_map[y_2, x_2] != WALL
                and (y_1, x_1) != (y_2, x_2))

    def step(self, actions):
        rewards = [0.0, 0.0]
        new_agent_coords = np.copy(self.current_agents_coords)

        for agent_idx, action in enumerate(actions):
            if self.move_completed[agent_idx]:
                continue

            dy, dx = ACTION_EFFECTS[action]
            y, x = self.current_agents_coords[agent_idx]
            new_y, new_x = y + dy, x + dx

            # Check for illegal moves
            if not self._within_bounds(new_y, new_x) or self.current_grid_map[new_y, new_x] == WALL:
                rewards[agent_idx] = -0.1
                continue

            if self._target_reached(agent_idx, new_y, new_x):
                self.move_completed[agent_idx] = True
                rewards[agent_idx] = 100.0
                self.current_grid_map[new_y, new_x] = SUCCESS
            else:
                rewards[agent_idx] = -0.1
                new_agent_coords[agent_idx] = [new_y, new_x]

        self._update_grid_map(self.current_agents_coords, new_agent_coords)
        self.current_agents_coords = new_agent_coords

        # Update game state and check if episode is done
        done = all(self.move_completed)
        return self._get_obs(), rewards, done

    def reset(self):
        self.current_grid_map = np.copy(self.start_grid_map)
        self.current_agents_coords = np.copy(self.agents_start_coords)
        self.move_completed = [False, False]
        self.episode_total_reward = 0.0
        return self._get_obs()

    def render(self, mode='human'):
        img = self._gridmap_to_image()
        if mode == 'rgb_array':
            return img
        elif mode == 'human':
            plt.imshow(img)
            plt.show()

    def _within_bounds(self, y, x):
        return 0 <= y < self.current_grid_map.shape[0] and 0 <= x < self.current_grid_map.shape[1]

    def _target_reached(self, agent_idx, y, x):
        target = TARGET1 if agent_idx == 0 else TARGET2
        return self.current_grid_map[y, x] == target

    def _find_agents_and_targets(self):
        start_coords = []
        target_coords = []
        for agent, target in [(AGENT1, TARGET1), (AGENT2, TARGET2)]:
            sy, sx = np.where(self.start_grid_map == agent)
            ty, tx = np.where(self.start_grid_map == target)
            start_coords.append([sy[0], sx[0]])
            target_coords.append([ty[0], tx[0]])
        return np.array(start_coords), np.array(target_coords)

    def _get_obs(self):
        return self.current_grid_map

    def _update_grid_map(self, old_coords, new_coords):
        # self.current_grid_map = np.copy(self.start_grid_map)

        old_grid_map = copy.deepcopy(self.current_grid_map)

        for (y, x) in old_coords:
            self.current_grid_map[y, x] = EMPTY

        for idx, (y, x) in enumerate(new_coords):
            self.current_grid_map[y, x] = AGENT1 if idx == 0 else AGENT2

        if new_coords[0][0] == self.agents_target_coords[0][0] and new_coords[0][1] == self.agents_target_coords[0][1]:
            self.current_grid_map[new_coords[0][0], new_coords[0][1]] = SUCCESS
        if new_coords[1][0] == self.agents_target_coords[1][0] and new_coords[1][1] == self.agents_target_coords[1][1]:
            self.current_grid_map[new_coords[1][0], new_coords[1][1]] = SUCCESS

        # In case an agent is on the other agent's target
        for i, (y, x) in enumerate(new_coords):
            if self.current_grid_map[y, x] == AGENT1 and (old_grid_map[y, x] == TARGET2 or old_grid_map[y, x] == TARGET2_OC1):
                self.current_grid_map[y, x] = TARGET2_OC1

            if self.current_grid_map[y, x] == AGENT2 and (old_grid_map[y, x] == TARGET1 or old_grid_map[y, x] == TARGET1_OC2):
                self.current_grid_map[y, x] = TARGET1_OC2

        # Check if agent moved away from occupying the foreign target
        if old_grid_map[old_coords[1][0], old_coords[1][1]] == TARGET1_OC2 and (new_coords[1][0] != old_coords[1][0] or new_coords[1][1] != old_coords[1][1]):
            self.current_grid_map[old_coords[1][0], old_coords[1][1]] = TARGET1

        if old_grid_map[old_coords[0][0], old_coords[0][1]] == TARGET2_OC1 and (new_coords[0][0] != old_coords[0][0] or new_coords[0][1] != old_coords[0][1]):
            self.current_grid_map[old_coords[0][0], old_coords[0][1]] = TARGET2

    def _gridmap_to_image(self):
        img = np.zeros((*self.current_grid_map.shape, 3))
        for i in range(self.current_grid_map.shape[0]):
            for j in range(self.current_grid_map.shape[1]):
                img[i, j] = COLORS[self.current_grid_map[i, j]]
        return img

    def _read_grid_map(self, grid_map_path):
        with open(grid_map_path, 'r') as file:
            grid_map = [[int(cell) for cell in line.split()] for line in file]
        return np.array(grid_map, dtype=int)


if __name__ == "__main__":
    env = GridworldEnv('1', from_file=True)
    obs = env.reset()
    done = False
    counter = 0

    while not done:
        counter += 1
        actions = env.get_legal_action_pairs()[random.randint(0, len(env.get_legal_action_pairs()) - 1)]
        obs, rewards, done = env.step(actions)

    env.render()

    #     #obs, rewards, done = env.step(actions)
    #     #if done:
    #     #    env.render()
    #     if r < 0.0001:
    #         print(r)
    #         env.render()
