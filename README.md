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

learning curve: 

<img src="https://github.com/qihongl/demo-q-learning/blob/master/imgs/lc.png" alt="lc" height=400px>

<br>

a sample path (red dot = reward, black dot = bomb): 

<img src="https://github.com/qihongl/demo-q-learning/blob/master/imgs/path.png" alt="path" height=400px>
