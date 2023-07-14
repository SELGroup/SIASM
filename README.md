# ACORN
## Paper
"**Socialbots on Fire: Modeling Adversarial Behaviors of Socialbots via Multi-Agent Hierarchical Reinforcement Lear**ning.", Web Conference (WWW) 2022. [arxiv](https://arxiv.org/abs/2110.10655)

Many of utils codes are credited to https://dl.acm.org/doi/10.5555/3398761.3398831 at https://github.com/kage08/graph_sample_rl
Many of the current codes are in very raw forms. Please use with caution.

## Specification of dependencies
- Python version ``3.8``
- Check ``req.txt`` file for details. Basically, we will need ``torch``, ``ray[rllib]``, ``tensorflow``, ``networkx``, and other basic packages. All other libraries and their version are stored in ``req.txt`` file. Or you can install all of the libraries by running:
```
conda create --name <env> --file req.txt
``` 
- Install the ``gym_bot`` environment:  
```
cd gym_bot
python -m pip install -e .
```
- The main ``gym`` environment file is at ``./gym-bot/gym_bot/envs/advbot_env_single_detect_large_hiar.py``.

## Dataset
- All the collected 100 news propagation networks are stored in ``./database/_hoaxy#.pkl`` with ``#`` ranges from 0-99. These are ``networkx`` python object for convenient loading with ``networkx`` library.
- The test script (below) will automatically load the ``train`` and ``test`` portion of the dataset.

## BlackBox Bot Detector
- We also provide the ``blackbox`` bot detector trained on the configurations specified in the paper at ``./detector/RandomForestClassifier_TRAM_lengthNone.joblib``. This is a scikit-learn ``Random Forest Classifier`` object that can be loaded using ``pickle`` library.
- Please refer to ``./gym-bot/gym_bot/envs/advbot_env_single_detect_large_hiar.py`` to how to load and use this bot detector.

## Hyper-Parameters and Model's Settings
Check the ``ppo_single_large_hiar.py``.
```
config = {
    "NAME":'advbot-v6',
    "run_name":None, 
    "seed":SEED, 
    "probs":0.8, #set -1 to random
    "graph_algorithm":"node2vec", 
    "WALK_P":1, # parameter p of node2vec
    "WALK_Q":50, # parameter q of node2vec
    "model_type":"CONV", 
    "node_embed_dim":6, # node embedding dimension of node2vec
    "num_filters":8, # number of filters for CONV
    "validation_graphs":[],
    "reward_shaping":None, 
    "num_workers":5, # number of workers used during train/test
    "num_gpus":1, # number of GPUS
    "graph_feature":"gcn", # gcn means node2vec features
    "lr":0.0003, # learning rate
    "entropy_coeff":0.01, # ppo parameter
    "training_iteration":10000, # number of training iterations
    "checkpoint_freq":5, # frequency of saving checkpoints during training
    "wandb_key":"" #wandb API (replace with your own)
}

config_test = {
    "custom_max_step": 120, # we train on 60 timesteps be default but during test we test on longer 120
    "detection_interval":20, # interval K refered in the paper
    "greedy": False, # whether test the AgentI+H in the paper (heuristic method)
}
```

## Train
RUN: ``python ppo_single_large_hiar.py train``
Example of Statistics on Synthetic Graphs. 
![Statistics on Synthetic Graphs](https://raw.githubusercontent.com/lethaiq/ACORN/main/resources/synthetic.png?token=ADJNWYT7SR4MDZULGAGCUHDAXUWJQ)


## Test from Pre-trained Model
- The checkpoint ``./checkpoint_best/checkpoint-150`` is the best checkpoint, **result of which is resulted in the paper.**  
- To reproduce the reuslts, run: ``python ppo_single_large_hiar.py test ./checkpoint_best/checkpoint-150`` for ``ACORN`` and ``python ppo_single_large_hiar.py greedy ./checkpoint_best/checkpoint-150`` for ``AgentI+H`` baseline. Please change the configuration in the ``ppo_single_large_hiar.py`` file for testing with specific parameters such as ``p`` (network actionvation probability), ``custom_max_length`` (maximum time horizon T), etc.

- Example outputs:
```
...
GRAPH: ./database/_hoaxy36.pkl
updating INTERVAL to  20
EVALUATING REAL GRPAH... 1500
[Parallel(n_jobs=2)]: Using backend LokyBackend with 2 concurrent workers.
[Parallel(n_jobs=2)]: Done   2 out of   2 | elapsed:    0.0s remaining:    0.0s
[Parallel(n_jobs=2)]: Done   2 out of   2 | elapsed:    0.0s finished
EVALUATING REAL GRPAH... 1500
[Parallel(n_jobs=2)]: Using backend LokyBackend with 2 concurrent workers.
[Parallel(n_jobs=2)]: Done   2 out of   2 | elapsed:    0.0s remaining:    0.0s
[Parallel(n_jobs=2)]: Done   2 out of   2 | elapsed:    0.0s finished
DONE HERE 120 120
out_degree [1 3 2 1 1] 120
Action Sequence (First 10, Last 10): MAMMRTMMRA MAMMRMAMMR
Number of Interaction: 124
Reward: 1.002000250031254
...
```

## Citation
```
@article{acorn2022,
    title={Socialbots on Fire: Modeling Adversarial Behaviors of Socialbots via Multi-Agent Hierarchical Reinforcement Learning},
    author={Thai Le and Long-Thanh Tran and Dongwon Lee},
    year={2022},
    journal={Proceedings of the 31st ACM Web Conference 2022 (WWW'20)},
}
```
