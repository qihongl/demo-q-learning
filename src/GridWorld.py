'''a grid world
adpated from: https://github.com/michaeltinsley/Gridworld-with-Q-Learning-Reinforcement-Learning-
'''
import numpy as np

ACTIONS = ['UP', 'DOWN', 'LEFT', 'RIGHT']
SIDE_LEN = 5


class GridWorld:
    # Initialise starting data
    def __init__(self):
        # Set information about the gridworld
        self.height = self.width = SIDE_LEN
        self.n_locs = self.height*self.width
        # Set locations for the bomb and the gold
        self.gold_loc = np.array([0, 3])
        self.bomb_loc = np.array([1, 3])
        self.terminal_states = [self.bomb_loc, self.gold_loc]
        # define the state->reward function
        self.SR_func = np.zeros((self.height, self.width)) - 1
        self.SR_func[tuple(self.bomb_loc)] = -10
        self.SR_func[tuple(self.gold_loc)] = 10
        # init agent location
        self.reset()

    def reset(self):
        self.cur_loc = np.array(
            [self.height-1, np.random.randint(0, self.width)])

    def is_terminal(self):
        return np.any([
            np.all(self.cur_loc == ts) for ts in self.terminal_states
        ])

    def get_agent_loc(self):
        '''get the current location / state of the agent'''
        grid = np.zeros((self.height, self.width))
        grid[tuple(self.cur_loc)] = 1
        return grid

    def get_reward(self, input_loc):
        '''compute the reward given the current state'''
        return self.SR_func[tuple(input_loc)]

    def step(self, action_):
        """take an action, update the current location of the agent, and
        return a reward

        Parameters
        ----------
        action : int
            [0,1,2,3] -> ['UP', 'DOWN', 'LEFT', 'RIGHT']

        Returns
        -------
        a number
            the reward at time t

        """
        action = ACTIONS[action_]
        if action == 'UP':
            if self.cur_loc[0] != 0:
                self.cur_loc += [-1, 0]
        elif action == 'DOWN':
            if self.cur_loc[0] != self.height - 1:
                self.cur_loc += [1, 0]
        elif action == 'LEFT':
            if self.cur_loc[1] != 0:
                self.cur_loc += [0, -1]
        elif action == 'RIGHT':
            if self.cur_loc[1] != self.width - 1:
                self.cur_loc += [0, 1]
        else:
            raise ValueError(f'unrecognizable action')
        # compute return info
        r_t = self.get_reward(self.cur_loc)
        return r_t


def construct_loc(loc_h, loc_w, height=SIDE_LEN, width=SIDE_LEN):
    grid = np.zeros((height, width))
    grid[tuple([loc_h, loc_w])] = 1
    return grid
