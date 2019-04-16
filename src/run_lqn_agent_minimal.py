'''run a linear q learning network on a grid world
do the weight update by hand (w/o any autodiff machinery) for edu purpose
'''

from GridWorld import GridWorld, ACTIONS
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import seaborn as sns
sns.set(style='white', context='talk', palette='colorblind')
np.random.seed(0)

# define env and agent
env = GridWorld()
state_dim = env.height * env.width
n_actions = len(ACTIONS)
# weights (i.e. the agent)
W = np.zeros((state_dim, n_actions))

# training params
n_trials = 150
max_steps = 50
epsilon = 0.2
alpha = 0.1
gamma = .9


'''train
'''
log_return = []
log_steps = []
log_actions = []
log_states = []

for i in range(n_trials):

    env.reset()
    cumulative_reward = 0
    step = 0
    log_actions_i = []
    log_states_i = []

    while step < max_steps:
        # get current state
        s_t = env.get_agent_loc().reshape(1, -1)
        log_states_i.append(s_t)
        # compute q val
        q_t = np.dot(s_t, W)
        # epsilon greedy action selection
        if np.random.uniform() > epsilon:
            a_t = np.argmax(q_t)
        else:
            a_t = np.random.randint(n_actions)
        # transition and get reward
        r_t = env.step(a_t)
        # get next states info
        s_next = env.get_agent_loc().reshape(1, -1)
        max_q_next = np.max(np.dot(s_next, W))
        # compute TD target
        q_expected = r_t + gamma * max_q_next

        # update weights
        w_delta = alpha * (q_expected) * s_t
        W[:, a_t] += np.squeeze(w_delta)

        # update R and n steps
        step += 1
        cumulative_reward += r_t * gamma**step
        log_actions_i.append(a_t)

        # termination condition
        if env.is_terminal():
            break

    log_states_i.append(s_t)

    log_states.append(log_states_i)
    log_actions.append(log_actions_i)
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
axes[1].axhline(0, color='grey', linestyle='--')
axes[1].set_ylabel('n steps taken')
axes[1].set_xlabel('Epoch')
axes[1].set_ylim([0, None])

sns.despine()
f.tight_layout()

'''weights'''

f, axes = plt.subplots(4, 1, figsize=(5, 11))
for i, ax in enumerate(axes):
    sns.heatmap(
        W[:, i].reshape(5, 5),
        cmap='viridis', square=True,
        vmin=np.min(W), vmax=np.max(W),
        ax=ax
    )
    ax.set_title(ACTIONS[i])
f.tight_layout()


'''show a sample trajectory'''

env.reset()
cumulative_reward = 0
step = 0
locs = []
locs.append(env.get_agent_loc())

while step < max_steps:
    # get an input
    s_t = locs[step].reshape(1, -1)
    q_t = np.dot(s_t, W)
    r_t = env.step(np.argmax(q_t))
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
