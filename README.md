# demo-q-learning

a implementation of q learning from scratch

### files: 
```
src/
├── GridWorld.py                    # the env
├── LinearAgent.py                  # a linear agent in pytorch 
├── run_lqn_agent.py                # run the pytorch linear agent 
├── run_lqn_agent_minimal.py        # run a linear agent, update weights by hand (no autodiff)
└── utils.py
```

### results: 

here's the q learning update rule, the agent is also epsilon greedy 

<img src="https://wikimedia.org/api/rest_v1/media/math/render/svg/47fa1e5cf8cf75996a777c11c7b9445dc96d4637" alt="lc" height=100px> 

here's the learning curve from one agent: 

<img src="https://github.com/qihongl/demo-q-learning/blob/master/imgs/lc.png" alt="lc" height=400px>

<br>

here's a sample path from a trained agent; red dot = reward, black dot = bomb: 

<img src="https://github.com/qihongl/demo-q-learning/blob/master/imgs/path.png" alt="path" height=350px>
