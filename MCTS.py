import copy

import numpy as np
from scipy.stats import sem
import random

from gridworld_env import GridworldEnv
from numba import jit

def state_to_env(state, plan=1):
    """
    The state is represented as

    (y1, x1, y2, x2, prev_act1, done1, prev_act2, done2)
    """

    new_env = GridworldEnv(1)

    return



class Node:
    def __init__(self, state, parent=None):
        self.state = copy.deepcopy(state)
        self.parent = parent
        self.children = []
        self.visits = 0
        self.value = 0

    def ucb1(self, total_visits, c_param=1.41):
        if self.visits == 0:
            return float('inf')  # Encourage exploration of unvisited nodes
        return self.value / self.visits + c_param * np.sqrt(np.log(total_visits) / self.visits)

class MCTS:
    def __init__(self, env, num_iterations, max_depth, agent_idx=0):
        self.env = env
        self.num_iterations = num_iterations
        self.max_depth = max_depth
        self.agent_idx = 0

    def select_node(self, node):
        while node.children:
            visits = sum(child.visits for child in node.children)
            node = max(node.children, key=lambda x: x.ucb1(visits))
        return node

    def expand(self, node):
        if node.visits == 0:  # Simulate from this node if it's the first visit
            return node

        # Otherwise, expand child nodes for each possible action
        for actions in node.state.get_legal_action_pairs():
            new_env = copy.deepcopy(node.state)
            new_env.step(actions)  # Assuming deterministic environment for simplicity
            node.children.append(Node(new_env, node))
        return random.choice(node.children)  # Return a random child node

    def simulate(self, node):
        current_depth = 0
        env_copy = copy.deepcopy(node.state)
        total_cum_reward = 0
        while current_depth < self.max_depth:
            sampled_actions = random.choice(env.get_legal_action_pairs())
            _, reward, agent_at_goal = env_copy.step(sampled_actions)
            if agent_at_goal[0]:
                break
            current_depth += 1

            if not node.state.move_completed[0] and not node.state.move_completed[1]:
                total_cum_reward += reward[0] + reward[1]
            elif node.state.move_completed[0] and not node.state.move_completed[1]:
                total_cum_reward += reward[1]
            elif not node.state.move_completed[0] and node.state.move_completed[1]:
                total_cum_reward += reward[0]

        return total_cum_reward

    def backpropagate(self, node, reward):
        while node is not None:
            node.visits += 1
            node.value += reward
            node = node.parent

    def run(self, initial_state):
        root = Node(initial_state)

        for _ in range(self.num_iterations):
            leaf = self.select_node(root)
            new_child = self.expand(leaf)
            reward = self.simulate(new_child)
            self.backpropagate(new_child, reward)

        # After running simulations, choose the action of the best performing child of the root
        return max(root.children, key=lambda x: x.value / x.visits if x.visits > 0 else float('-inf')).state

@jit(nopython=True)
def move_to_action(move):
    if move == (0, 0):
        return 0
    elif move == (-1, 0):
        return 1
    elif move == (1, 0):
        return 2
    elif move == (0, -1):
        return 3
    elif move == (0, 1):
        return 4


def get_actions_from_env_diff(old_env, new_env):
    state = old_env.current_game_state
    next_state = new_env.current_game_state

    # Calculate differences in positions
    dy1 = next_state[0] - state[0]
    dx1 = next_state[1] - state[1]
    dy2 = next_state[2] - state[2]
    dx2 = next_state[3] - state[3]

    return move_to_action((dy1, dx1)), move_to_action((dy2, dx2))


if __name__ == '__main__':
    env = GridworldEnv(1)
    env.reset()
    done = False
    runner_env = copy.deepcopy(env)

    counter = 0
    while not done:
        counter += 1
        game = MCTS(runner_env, 100, max_depth=1000)
        next_env = game.run(runner_env)

        actions = get_actions_from_env_diff(runner_env, next_env)
        runner_env.step(actions)
        if counter % 10 == 0:
            runner_env.render()

        print(runner_env.move_completed)
        done = all(runner_env.move_completed)
        print(runner_env.get_legal_action_pairs())

    print('done')
