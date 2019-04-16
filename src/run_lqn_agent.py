'''run a linear q network on a grid world'''

from envs.GridWorld import GridWorld, ACTIONS
from agent.Linear import Linear as Agent
from utils import to_torch

import numpy as np
import os
import torch
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
env = GridWorld(has_bomb=True)
state_dim = env.height * env.width
n_actions = len(ACTIONS)
agent = Agent(state_dim, n_actions)

# training params
n_trials = 300
max_steps = 50
epsilon = .3
alpha = 0.3
gamma = .8
optimizer = optim.SGD(agent.parameters(), lr=alpha)

'''train
'''
log_return = []
log_steps = []
for i in range(n_trials):
    env.reset()
    cumulative_reward = 0
    step = 0

    while step < max_steps:
        # get current state
        s_t = to_torch(env.get_agent_loc().reshape(1, -1))
        # compute q val
        q_t = agent.forward(s_t)
        # epsilon greedy action selection
        if np.random.uniform() > epsilon:
            a_t = torch.argmax(q_t)
        else:
            a_t = np.random.randint(n_actions)
        # transition and get reward
        r_t = env.step(a_t)
        # get next states info
        s_next = to_torch(env.get_agent_loc().reshape(1, -1))
        max_q_next = torch.max(agent.forward(s_next))
        # compute TD target
        q_target = r_t + gamma * max_q_next

        # update weights
        loss = F.smooth_l1_loss(q_t[:, a_t], q_target.data)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # update R and n steps
        step += 1
        cumulative_reward += r_t * gamma**step

        # termination condition
        if env.is_terminal():
            break

    log_return.append(cumulative_reward)
    log_steps.append(step)

'''
learning curve
'''

f, axes = plt.subplots(2, 1, figsize=(6, 6), sharex=True)

axes[0].plot(log_return)
axes[0].axhline(0, color='grey', linestyle='--')
axes[0].set_title('Learning curve')
axes[0].set_ylabel('Return')

axes[1].plot(log_steps)
axes[1].set_title(' ')
axes[1].set_ylabel('n steps taken')
axes[1].set_xlabel('Epoch')
axes[1].set_ylim([0, None])

sns.despine()
f.tight_layout()
f.savefig(os.path.join(img_dir, 'lqn-lc.png'), dpi=120)

'''weights'''

W = agent.linear.weight.data.numpy()
# vmin =
f, axes = plt.subplots(4, 1, figsize=(5, 11))
for i, ax in enumerate(axes):
    sns.heatmap(
        W[i, :].reshape(5, 5),
        cmap='viridis', square=True,
        vmin=np.min(W), vmax=np.max(W),
        ax=ax
    )
    ax.set_title(ACTIONS[i])
f.tight_layout()
f.savefig(os.path.join(img_dir, 'lqn-wts.png'), dpi=120)

'''show a sample trajectory'''
env.reset()
cumulative_reward = 0
step = 0
locs = []
locs.append(env.get_agent_loc())

while step < max_steps:
    # get an input
    s_t = to_torch(locs[step].reshape(1, -1))
    q_t = agent.forward(s_t)
    r_t = env.step(torch.argmax(q_t))
    step += 1
    cumulative_reward += r_t * gamma**step
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
f.savefig(os.path.join(img_dir, 'lqn-path.png'), dpi=120)
