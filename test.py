import os
import ray
import time
import numpy as np

from gym_bot.envs import AdvBotEnvSingleDetectLargeHiar
from models import TorchParametricActionsModel
from ray import tune
from ray.rllib.agents.ppo import PPOTrainer
from ray.rllib.models import ModelCatalog
from ray.tune import run_experiments
from ray.tune.integration.wandb import WandbLoggerCallback
from ray.tune.registry import register_env


def test(
    model_path,
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
    training_iteration=10000,
    checkpoint_freq=5,
    custom_max_step=120,
    detection_interval=20,
    greedy=False,
    wandb_key=None):

    ray.init()
    def env_creator(graphs=[]):
        env = AdvBotEnvSingleDetectLargeHiar(seed=seed, 
                                        validation=True,
                                        validation_graphs=graphs,
                                        graph_algorithm=graph_algorithm.lower(),
                                        walk_p=WALK_P,
                                        walk_q=WALK_Q,
                                        model_type=model_type,
                                        node_embed_dim=node_embed_dim,
                                        probs=probs,
                                        graph_feature=graph_feature,
                                        custom_max_step=custom_max_step,
                                        interval=detection_interval)
        return env

    register_env(NAME, env_creator)
    ModelCatalog.register_custom_model("pa_model", TorchParametricActionsModel)
    env = env_creator()

    act_dim = env.action_dim
    obs_dim = env.level2_observation_space['advbot'].shape
    activated_dim = env.level2_observation_space['activated'].shape
    history_dim = env.level2_observation_space['history'].shape

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
    policy_graphs['level1'] = (None, env.level1_observation_space, env.level1_action_space, {})
    policy_graphs['level2'] = (None, env.level2_observation_space, env.level2_action_space, level2_model_config)

    def policy_mapping_fn(agent_id):
        return agent_id

    config={
        "log_level": "WARN",
        "num_workers": num_workers,
        "num_gpus": num_gpus,
        "multiagent": {
            "policies": policy_graphs,
            "policy_mapping_fn": policy_mapping_fn,
        },
        "seed": seed + int(time.time()),
        'framework': 'torch',
        "env": NAME
    }

    agent = None
    agent = PPOTrainer(config=config, env=NAME)
    agent.restore(model_path)
    print("RESTORED CHECKPOINT")

    def get_action(obs, agent=None, env=None, greedy=None):
        action = {}
        if not greedy:
            explores = {
                'level1': False,
                'level2': False
            }
            action = {}
            for agent_id, agent_obs in obs.items():
                policy_id = config['multiagent']['policy_mapping_fn'](agent_id)
                action[agent_id] = agent.compute_action(agent_obs, policy_id=policy_id, explore=explores[agent_id])

        else: #greedy
            assert env != None, "Need to provide the environment for greedy baseline"
            for agent_id, agent_obs in obs.items():
                if agent_id == "level1":
                    policy_id = config['multiagent']['policy_mapping_fn'](agent_id)
                    action[agent_id] = agent.compute_action(agent_obs, policy_id=policy_id, explore=False)
                else:
                    action[agent_id] = env.next_best_greedy()

        return action

    total_rewards = []
    total_ts = []

    for name in validation_graphs:
        print("\nGRAPH: {}".format(name))
        graph = validation_graphs[name]
        env = env_creator(graphs=[graph])
        count = {}
        done = False
        obs = env.reset()
        while not done:
            action = get_action(obs, agent, env=env, greedy=greedy)
            obs, reward, done, info = env.step(action)
            done = done['__all__']

        seeds = list(env.seed_nodes.keys())
        reward = env.cal_rewards(test=True, seeds=seeds, reward_shaping=None)
        reward = 1.0 * reward/env.best_reward()
        total_rewards.append(reward)
        total_ts.append(env.current_t)

        print("Action Sequence (First 10, Last 10):", env.state[:10], env.state[-10:])
        print("Number of Interaction:", len(env.state) - env.state.count("T"))
        print("Reward:", reward)

    print(total_rewards, np.mean(total_rewards), np.std(total_rewards))
    print(total_ts, np.mean(total_ts), np.std(total_ts))
