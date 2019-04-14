'''a grid world

# Classes for the Enviroment and the Agent

- The GridWorld class contains the environment
- The dimensions of the environment are defined
- Locations of all rewards are stored

- Functions for different methods written
    - `get_available_actions` returns possible actions
    - `get_agent_loc` prints out current location of the agent on the grid
    - `get_reward` returns the reward for an input position
    - `make_step` moves the agent in a specified direction

'''
import numpy as np

ACTIONS = ['UP', 'DOWN', 'LEFT', 'RIGHT']
SIDE_LEN = 5


class GridWorld:
    # Initialise starting data
    def __init__(self):
        # Set information about the gridworld
        self.height = self.width = SIDE_LEN
        # set default r for all states
        self.grid = np.zeros((self.height, self.width)) - 1
        # Set locations for the bomb and the gold
        self.gold_location = (0, 3)
        self.bomb_location = (1, 3)
        self.grid[self.bomb_location] = -10
        self.grid[self.gold_location] = 10
        #
        self.terminal_states = [self.bomb_location, self.gold_location]
        self.reset()

    def reset(self):
        self.cur_loc = (SIDE_LEN-1, np.random.randint(0, SIDE_LEN))

    def get_agent_loc(self):
        """Prints out current location of the agent on the grid"""
        grid = np.zeros((self.height, self.width))
        grid[self.cur_loc[0], self.cur_loc[1]] = 1
        return grid

    def get_reward(self, new_location):
        """Returns the reward for an input position"""
        return self.grid[new_location[0], new_location[1]]

    def make_step(self, action):
        """Moves the agent in the specified direction.
        If agent is at a border, agent stays still
        but takes negative reward.
        Function returns the reward for the move."""
        # Store previous location
        prev_loc = self.cur_loc

        # UP
        if action == 'UP':
            # If agent is at the top, stay still, collect reward
            if prev_loc[0] == 0:
                reward = self.get_reward(prev_loc)
            else:
                self.cur_loc = (self.cur_loc[0] - 1, self.cur_loc[1])
                reward = self.get_reward(self.cur_loc)

        # DOWN
        elif action == 'DOWN':
            # If agent is at bottom, stay still, collect reward
            if prev_loc[0] == self.height - 1:
                reward = self.get_reward(prev_loc)
            else:
                self.cur_loc = (self.cur_loc[0] + 1, self.cur_loc[1])
                reward = self.get_reward(self.cur_loc)

        # LEFT
        elif action == 'LEFT':
            # If agent is at the left, stay still, collect reward
            if prev_loc[1] == 0:
                reward = self.get_reward(prev_loc)
            else:
                self.cur_loc = (self.cur_loc[0], self.cur_loc[1] - 1)
                reward = self.get_reward(self.cur_loc)

        # RIGHT
        elif action == 'RIGHT':
            # If agent is at the right, stay still, collect reward
            if prev_loc[1] == self.width - 1:
                reward = self.get_reward(prev_loc)
            else:
                self.cur_loc = (self.cur_loc[0], self.cur_loc[1] + 1)
                reward = self.get_reward(self.cur_loc)

        else:
            raise ValueError(f'unrecognizable action')

        return reward

    def is_terminal(self):
        """Check if the agent is in a terminal state (gold or bomb),
        if so return 'TERMINAL'
        """
        if self.cur_loc in self.terminal_states:
            return True
        return False
