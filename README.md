# SIASM
## Paper
"**Adversarial Socialbots Modeling Based on Structural Information Principles."

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
    "node_embed_dim":42, # node embedding dimension of node2vec
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
    "wandb_key":"" #wandb API (replace with your own),
    "sip": True
}

config_test = {
    "custom_max_step": 120, # we train on 60 timesteps be default but during test we test on longer 120
    "detection_interval":20, # interval K refered in the paper
    "greedy": False, # whether test the AgentI+H in the paper (heuristic method)
}
```

## Train
RUN: ``python ppo_single_large_hiar.py train``


## Test from Pre-trained Model  
- To reproduce the reuslts, run: ``python ppo_single_large_hiar.py test ./checkpoint_best/checkpoint`` for ``SIASM`` and ``python ppo_single_large_hiar.py greedy ./checkpoint_best/checkpoint`` for ``SIASM-H`` baseline. 
