'''run a linear q learning network on a grid world'''

from envs.GridWorld import GridWorld, ACTIONS
from utils import to_torch

import numpy as np
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import seaborn as sns

sns.set(style='white', context='talk', palette='colorblind')
np.random.seed(0)
torch.manual_seed(0)

img_dir = '../imgs'

# define env and agent
env = GridWorld(side_len=5, has_bomb=True)
# the agent
state_dim = env.height * env.width
n_actions = len(ACTIONS)
dim_hidden = 4
rnn = nn.LSTM(state_dim, dim_hidden)
readout = nn.Linear(dim_hidden, n_actions)
all_params = list(rnn.parameters())+list(readout.parameters())


'''train
'''


def get_init_state():
    h_0 = torch.randn(1, 1, dim_hidden)
    c_0 = torch.randn(1, 1, dim_hidden)
    return (h_0, c_0)


# training params
n_trials = 500
max_steps = 50
epsilon = .3
alpha = 0.1
gamma = .9
optimizer = optim.RMSprop(all_params, lr=alpha)

# prealloc
log_return = []
log_steps = []
for i in range(n_trials):

    env.reset()
    cumulative_reward = 0
    step = 0

    while step < max_steps:
        if step == 0:
            h_prev = get_init_state()

        # get current state to predict action value
        s_t = to_torch(env.get_agent_loc().reshape(1, -1))
        out_t, h_t = rnn(s_t.view(1, 1, -1), h_prev)
        q_t = readout(out_t)

        # epsilon greedy action selection
        if np.random.uniform() > epsilon:
            a_t = torch.argmax(q_t)
        else:
            a_t = np.random.randint(n_actions)
        # transition and get reward
        r_t = env.step(a_t)

        # get next states info
        s_next = to_torch(env.get_agent_loc().reshape(1, -1))
        out_next, _ = rnn(s_next.view(1, 1, -1), h_t)
        q_next = readout(out_next)
        # compute TD target
        q_target = r_t + gamma * torch.max(q_next)

        # update weights
        loss = F.smooth_l1_loss(q_t[:, :, a_t], q_target.data)
        optimizer.zero_grad()
        loss.backward(retain_graph=True)
        optimizer.step()

        step += 1
        cumulative_reward += r_t * gamma**step
        h_prev = h_t

        # termination condition
        if env.is_terminal():
            break

    log_return.append(cumulative_reward)
    log_steps.append(step)
    if np.mod(i, 10) == 0:
        print('Epoch: %d | ns = %d, R = %.2f ' % (i, step, cumulative_reward))

'''
learning curve
'''
current_palette = sns.color_palette()
ws = 20
log_steps_smoothed = [
    np.mean(log_steps[k:k+ws]) for k in range(n_trials-ws+1)
]
log_return_smoothed = [
    np.mean(log_return[k:k+ws]) for k in range(n_trials-ws+1)
]

f, axes = plt.subplots(2, 1, figsize=(6, 6), sharex=True)
axes[0].plot(log_return, color=current_palette[0], alpha=.3)
axes[0].plot(log_return_smoothed)
axes[0].axhline(0, color='grey', linestyle='--')
axes[0].set_title('Learning curve')
axes[0].set_ylabel('Return')

axes[1].plot(log_steps, color=current_palette[0], alpha=.3)
axes[1].plot(log_steps_smoothed)
# axes[1].axhline(env.height-1, color='grey', linestyle='--')
axes[1].set_title(' ')
axes[1].set_ylabel('n steps taken')
axes[1].set_xlabel('Epoch')
axes[1].set_ylim([0, None])

sns.despine()
f.tight_layout()
f.savefig(os.path.join(img_dir, 'rqn-lc.png'), dpi=120)

'''show a sample trajectory'''

env.reset()
cumulative_reward = 0
step = 0
locs = []
locs.append(env.get_agent_loc())

while step < max_steps:
    if step == 0:
        h_prev = get_init_state()
    # get an input
    s_t = to_torch(locs[step].reshape(1, -1))
    out_t, h_t = rnn(s_t.view(1, 1, -1), h_prev)
    q_t = readout(out_t)
    r_t = env.step(torch.argmax(q_t))
    #
    step += 1
    cumulative_reward += r_t * gamma**step
    h_prev = h_t
    # log
    locs.append(env.get_agent_loc())
    if env.is_terminal():
        break
step += 1

color_intensity = np.linspace(.1, 1, step)
path = np.sum([color_intensity[t]*locs[t] for t in range(step)], axis=0)

f, ax = plt.subplots(1, 1, figsize=(5, 5))
ax.set_title(f'Steps taken = {step}; Return = %.2f' % cumulative_reward)
ax.imshow(path, cmap='Blues', aspect='auto')
goal = Circle(env.gold_loc[::-1], radius=.1, color='red')
ax.add_patch(goal)
if env.has_bomb:
    bomb = Circle(env.bomb_loc[::-1], radius=.1, color='black')
    ax.add_patch(bomb)
f.tight_layout()
f.savefig(os.path.join(img_dir, 'rqn-path.png'), dpi=120)
