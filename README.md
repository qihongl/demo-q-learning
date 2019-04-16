# demo-q-learning

some toy demos, q learning with neural network function approximator

### files: 
```
└── src
    ├── GridWorld.py              # the env
    ├── agent
    │   ├── Linear.py             # a linear network/regression 
    │   └── MLP.py                # a feed-forward network 
    ├── run_lqn_agent_minimal.py  # run a linear q network, update weights by hand (no autodiff)
    ├── run_lqn_agent.py          # run a linear q network     
    ├── run_mlp_agent.py          # run a feed-forward q network 
    ├── run_rnn_agent.py          # run a lstm q network 
    └── utils.py
```

### results: 

here's the q learning update rule, the agent is also epsilon greedy 

<img src="https://wikimedia.org/api/rest_v1/media/math/render/svg/47fa1e5cf8cf75996a777c11c7b9445dc96d4637" alt="lc" height=100px> 

here's the learning curve from one agent: 

<img src="https://github.com/qihongl/demo-q-learning/blob/master/imgs/rqn-lc.png" alt="lc" height=400px>

<br>

here's a sample path from a trained agent; red dot = reward, black dot = bomb: 

<img src="https://github.com/qihongl/demo-q-learning/blob/master/imgs/rqn-path.png" alt="path" height=350px>
