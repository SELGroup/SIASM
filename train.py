import os
import ray

from gym_bot.envs import AdvBotEnvSingleDetectLargeHiar
# from models import TorchParametricActionsModel
from siasm_model import TorchParametricActionsModel
from ray import tune
from ray.rllib.agents.ppo import PPOTrainer
from ray.rllib.models import ModelCatalog
from ray.tune import run_experiments
from ray.tune.integration.wandb import WandbLoggerCallback
from ray.tune.registry import register_env

def train(
    NAME='advbot-v6',
    run_name=None, 
    seed=77, 
    probs=-1,
    graph_algorithm="node2vec", 
    WALK_P=1, 
    WALK_Q=50, 
    model_type="CONV",
    node_embed_dim=6,
    num_filters=8,
    validation_graphs=[],
    reward_shaping=None,
    num_workers=5,
    num_gpus=1,
    graph_feature="gcn",
    lr=0.0003,
    entropy_coeff=0.01,
    training_iteration=1,
    checkpoint_freq=5,
    wandb_key=None,
    sip=True):

    def env_creator(_):
        env = AdvBotEnvSingleDetectLargeHiar(seed=seed, 
                                            validation=False,
                                            graph_algorithm=graph_algorithm.lower(),
                                            walk_p=WALK_P,
                                            walk_q=WALK_Q,
                                            model_type=model_type,
                                            node_embed_dim=node_embed_dim,
                                            probs=probs,
                                            graph_feature=graph_feature,
                                            validation_graphs=validation_graphs,
                                            reward_shaping=reward_shaping,
                                            sip=sip)
        return env

    register_env(NAME, lambda config: env_creator(None, **config))
    ModelCatalog.register_custom_model("pa_model", TorchParametricActionsModel)
    env = env_creator(None)
    act_dim = env.action_dim
    obs_dim = env.level2_observation_space['advbot'].shape
    activated_dim = env.level2_observation_space['activated'].shape if model_type == "CONV" else None
    history_dim = env.level2_observation_space['history'].shape if model_type == "CONV" else None

    level2_model_config = {
        "model": {
            "custom_model": "pa_model",
            "custom_model_config": {"model_type": model_type,
                                    "true_obs_shape": obs_dim, 
                                    "action_embed_size": act_dim,
                                    "node_embed_dim": node_embed_dim,
                                    "num_filters": num_filters,
                                    "activated_obs_shape": activated_dim,
                                    "history_obs_shape": history_dim},
            "vf_share_layers": True,
        }}

    level1_model_config = {}

    policy_graphs = {}
    policy_graphs['level1'] = (None, env.level1_observation_space, env.level1_action_space, level1_model_config)
    policy_graphs['level2'] = (None, env.level2_observation_space, env.level2_action_space, level2_model_config)

    def policy_mapping_fn(agent_id):
        return agent_id

    input_graphs = [validation_graphs[k] for k in validation_graphs]

    config={
        "log_level": "WARN",
        "num_workers": num_workers,
        "num_envs_per_worker": 1,
        "num_cpus_for_driver": 1,
        "num_cpus_per_worker": 1,
        "num_gpus": num_gpus,
        
        "multiagent": {
            "policies": policy_graphs,
            "policy_mapping_fn": policy_mapping_fn,
        },

        "lr": lr,
        "entropy_coeff": entropy_coeff,
        "seed": seed,
        'framework': 'torch',
        "env": NAME
    }

    exp_dict = {
        'name': 'hierachical_synthetic6',
        "local_dir": os.environ.get('ADVBOT_LOG_FOLDER'),
        'run_or_experiment': 'PPO',
        "stop": {
            "training_iteration": training_iteration
        },
        'checkpoint_freq':checkpoint_freq,
        "config": config,
        "callbacks": [WandbLoggerCallback(
            project="ACORN-{}".format(graph_algorithm),
            group="GraphSize-{}".format(act_dim-1),
            api_key=wandb_key,
            log_config=True)] if wandb_key else []
    }

    ray.init()
    tune.run(**exp_dict)