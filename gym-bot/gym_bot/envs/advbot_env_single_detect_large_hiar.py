import gym
import numpy as np
from scipy import sparse
import os
import warnings
import math
import networkx as nx

from gym import error
from gym import spaces
from gym import utils
from gym.utils import seeding
from joblib import dump
from joblib import load
import torch
import time
import glob
from gym.spaces import Box, Discrete, Dict
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from ray.rllib.models.preprocessors import get_preprocessor
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder
from torch.nn.utils.rnn import pad_sequence
from keras.utils.np_utils import to_categorical   
from ge.gcn_test import *
from ge.graph_utils import *
import scipy.stats as ss
from sip import *

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

class Detector:
    def __init__(self, model_path):
        self.scaler, self.model = load(model_path) 

    def predict(self, action, follower=None, following=None):
        x = self.extract_features(action, follower, following)
        x = self.scaler.transform(x)
        pred = self.model.predict(x)
        return pred

    def extract_features(self, action, follower=None, following=None):
        num_tweets = action.count('T')
        num_replies = action.count('A')
        num_retweets = action.count('R')
        num_mentions = action.count('M')
    # 
        avg_mentions_per_tweet = num_mentions / max(1, num_tweets)
        retweet_ratio = num_retweets / max(1, num_tweets)
        reply_ratio = num_replies / max(1, num_tweets)
        retweet_reply_ratio = num_retweets / max(1, reply_ratio)
        num_interactions = num_retweets + num_replies + num_mentions
        avg_interaction_per_tweet = num_interactions / max(1, num_tweets)

        rt = [num_tweets, num_replies, num_retweets]
        rt += [retweet_ratio, reply_ratio, retweet_reply_ratio]
        rt += [num_mentions, avg_mentions_per_tweet]
        rt += [num_interactions, avg_interaction_per_tweet]

        rt = np.array(rt).reshape(1, -1)
        return rt
        

class AdvBotEnvSingleDetectLargeHiar(MultiAgentEnv): #advbot-v6
# class AdvBotEnvSingleDetectLargeHiar(gym.Env): #advbot-v6
    metadata = {'render.modes': ['human']}
    ACTION = ["T", "R", "A", "M"]
    MAX_TIME_STEP = 60
    INTERVALS = [20, 40, 60, 80, 100, 120, 140, 160]
    INTERVAL = 20
    UPPER_TIME_LIMIT = 3000
    OUT_DEGREE_MIN = 0

    ORIGIN_GRAPH = "advbot_train"
    NUM_COMMUNITY = 100
    MODE = "out_degree"
    REWARD_SHAPING = None

    COMPRESS_FOLLOWSHIP = True
    VERBOSE = False

    def __init__(self, 
                num_bots=1, 
                discrete_history=False, 
                random_stimulation=True, 
                seed=77, 
                override={}, 
                validation=False,
                debug=False,
                graph_algorithm="node2vec",
                walk_p=1, 
                walk_q=1,
                flg_detection=True,
                model_type="FCN",
                node_embed_dim=2,
                probs=0.25,
                graph_feature="out_degree",
                custom_max_step=None,
                validation_graphs=[],
                reward_shaping=None,
                level1_independent=False,
                detector_type="RandomForest",
                interval=None,
                sip=True):
        self.seed(seed)

        for k in override:
            try:
                getattr(self, k)
                setattr(self, k, override[k])
                print("Update {} to {}".format(k, override[k]))
            except Exception as e:
                pass

        if interval:
            self.INTERVAL = interval
            print("updating INTERVAL to ", interval)

        if custom_max_step:
            self.MAX_TIME_STEP = custom_max_step
            self.INTERVALS = list(range(self.INTERVAL, self.MAX_TIME_STEP+1000*self.INTERVAL, self.INTERVAL))

        if reward_shaping:
            self.REWARD_SHAPING = reward_shaping


        self.MODEL_PATH = './detector/{}Classifier_TRAM_lengthNone.joblib'.format(detector_type)

        self.sip = sip
        self.level1_independent = level1_independent
        self.graph_algorithm = graph_algorithm
        self.walk_p = walk_p
        self.walk_q = walk_q
        self.node_embed_dim = node_embed_dim
        self.flg_detection = flg_detection
        self.PROB_RETWEET = probs
        self.MODE = graph_feature

        self.DEBUG = debug
        self.model_type = model_type
        self.validation = validation
        self.validation_graphs = validation_graphs
        self.discrete_history = discrete_history
        self.random_stimulation = random_stimulation

        self.n_fake_users = num_bots
        self.initialize()
        self.detector = Detector(self.MODEL_PATH)
        
        if self.VERBOSE:
            print("loaded bot detector", self.detector.model)


    def update_avail_actions(self):
        activated_idx = np.array(list(self.seed_nodes.keys()))
        self.action_mask = np.array([1] * self.max_avail_actions) ## all actions are available
        if len(activated_idx) > 0:
            self.action_mask[activated_idx] = 0
        # if self.sip and (self.current_t % 20) == 0:
        #     # user filtering based on the structural information principles
        #     print("user filtering based on the structural information principles")
        #     adj_matrix = np.zeros([len(self.G.nodes()), len(self.G.nodes())], dtype=np.float32)
        #     for edge in self.G.edges():
        #         weight = abs(np.corrcoef(self.G_obs[edge[0]], self.G_obs[edge[1]])[0, 1])
        #         adj_matrix[edge[0]][edge[1]] += weight
        #         adj_matrix[edge[1]][edge[0]] += weight
        #     pt = PartitionTree(adj_matrix=adj_matrix)
        #     pt.build_coding_tree(3)  # tree height
        #     ens = []
        #     ids = []
        #     for id in pt.tree_node.keys():
        #         node = pt.tree_node[id]
        #         if node.child_h == 1:
        #             ids.append(id)
        #             ens.append(pt.community_entropy(pt.tree_node, id, 0.0))
        #     disable_node = []
        #     idx = np.argsort(ens)
        #     vns = 0
        #     for i in range(len(idx)):
        #         id = ids[idx[i]]
        #         for vid in pt.tree_node[id].partition:
        #             disable_node.append(vid)
        #         vns += len(pt.tree_node[id].partition)
        #         if vns >= 0.01 * len(self.G.nodes()):
        #             break
        # else:
        #     print("user filtering based on local features")
        #     disable_node = np.where(self.out_degree <= self.OUT_DEGREE_MIN)[0]
        disable_node = np.where(self.out_degree <= self.OUT_DEGREE_MIN)[0]
        self.action_mask[disable_node] = 0

        if self.action_mask.sum() == 0: # if there is no valid action => open all
            self.action_mask = np.array([1] * self.max_avail_actions)
            if len(activated_idx) > 0:
                self.action_mask[activated_idx] = 0


    def vectorize_graph(self, g, mode="gcn"):
        if mode == "gcn":
            rt = np.stack(get_embeds(g, 
                    node_embed_dim=self.node_embed_dim,
                    alg=self.graph_algorithm,
                    p=self.walk_p, 
                    q=self.walk_q))

        elif mode == "out_degree":
            rt = self.out_degree/len(self.G)

        elif mode == "rank":
            rt = ss.rankdata(self.out_degree)/len(self.G)

        return rt


    def best_reward(self):
        idx = np.argsort(self.out_degree)[::-1][:self.MAX_TIME_STEP]
        cur_reward = self.compute_influence(self.G, list(idx), prob=self.PROB_RETWEET)
        return cur_reward


    def next_best_greedy(self):
        idx = np.argsort(self.out_degree)[::-1]
        for i in idx:
            if i not in self.seed_nodes:
                return i
        return np.random.choice(i)


    def initialize(self, reset_network=True):
        if self.PROB_RETWEET < 0:
            np.random.seed(int(str(time.time())[-5:]))
            self.PROB_RETWEET = np.random.choice([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
            print("SETTING PROB TWEET", self.PROB_RETWEET)

        if not (self.validation and len(self.validation_graphs)):
            self.G = randomize_graph(graph_name=self.ORIGIN_GRAPH, k=self.NUM_COMMUNITY, mode=3)
            for e in self.G.edges():
                self.G[e[0]][e[1]]['weight'] = self.PROB_RETWEET
        else:
            idx = np.random.choice(len(self.validation_graphs))
            self.G = self.validation_graphs[idx]
            print("EVALUATING REAL GRPAH...", len(self.G))

        self.out_degree = np.array([a[1] for a in list(self.G.out_degree(list(range(len(self.G)))))])
        # print("OUT_DEGREE", np.sort(self.out_degree)[::-1][:10])

        self.n_legit_users = len(self.G)
        self.max_avail_actions = self.n_legit_users
        self.state = ""
        self.seed_nodes = {}
        self.following = {}
        if not self.validation:
            self.current_interval = 0
        else:
            self.current_interval = 1

        # self.activated_idx = np.array([1]*self.n_legit_users)
        self.level1_reward = 0.0
        self.level2_reward = 0.0
        self.done = 0
        self.current_t = 0
        self.previous_reward = 0
        self.previous_rewards = []
        self.last_undetect = 0
        self.heuristic_optimal_reward = self.best_reward()
        self.action_mask = np.array([1] * self.max_avail_actions)

        self.G_obs = self.vectorize_graph(self.G, mode=self.MODE)
        
        self.action_dim = self.node_embed_dim
        random_state = np.random.RandomState(seed=7777)
        self.action_assignments = random_state.normal(0, 1, (self.max_avail_actions, self.action_dim)).reshape(self.max_avail_actions, self.action_dim)
        
        # if not self.validation:
        self.update_avail_actions()

        self.level1_action_space = gym.spaces.Discrete(len(self.ACTION))
        self.level1_observation_space = gym.spaces.Box(low=0, high=1, shape=self.pack_observation("level1").shape)
        self.level2_action_space = gym.spaces.Discrete(self.max_avail_actions)

        temp_obs = self.pack_observation("level2")
        if self.model_type == "FCN":
            self.level2_observation_space = Dict({
                "action_mask": Box(0, 1, shape=(self.max_avail_actions, )),
                "avail_actions": Box(-5, 5, shape=(self.max_avail_actions, self.action_dim)),
                "advbot":  gym.spaces.Box(low=-10, high=10, shape=temp_obs['advbot'].shape),
                "adjacency": gym.spaces.Box(low=0, high=1, shape=temp_obs['adjacency'].shape)
            })

        else:
            self.level2_observation_space = Dict({
                "action_mask": Box(0, 1, shape=(self.max_avail_actions, )),
                "avail_actions": Box(-5, 5, shape=(self.max_avail_actions, self.action_dim)),
                "advbot":  gym.spaces.Box(low=-100, high=100, shape=temp_obs['advbot'].shape),
                "activated": gym.spaces.Box(low=0, high=1, shape=temp_obs['activated'].shape),
                "history": gym.spaces.Box(low=0, high=1, shape=temp_obs['history'].shape),
                "adjacency": gym.spaces.Box(low=0, high=1, shape=temp_obs['adjacency'].shape)
            })

    def seed(self, seed=None):
        np.random.seed(seed)
        torch.manual_seed(seed)
        self.np_random, seed = seeding.np_random(seed)
        return [seed]


    def pack_observation(self, agent):
        state = self.state
        num_tweets = state.count('T')
        num_replies = state.count('A')
        num_retweets = state.count('R')
        num_mentions = state.count('M')
        bot_degree = len(self.following)

        if self.level1_independent:
            history = np.array([num_tweets, num_replies, num_retweets, num_mentions]).reshape(1, -1)
        else:
            history = np.array([num_tweets, num_replies, num_retweets, num_mentions, bot_degree]).reshape(1, -1)
        history = history / max(1, history.sum()) #dynamic

        if agent == "level1":
            obs = history
            # obs = np.concatenate((history, activated), 1)

        elif agent == "level2":
            network = self.G_obs
            activated_idx = np.array(list(self.seed_nodes.keys()))
            activated = np.array([0]*self.n_legit_users)
            if len(activated_idx):
                target_degree = np.array([a[1] for a in list(self.G.degree(activated_idx))])
                activated[activated_idx] = target_degree
            activated = activated.reshape(1, -1)
            activated = (1+len(self.following))/((1+activated))
            activated = activated / max(1, activated.sum())
            # graph to edge array
            adj_matrix = nx.adjacency_matrix(self.G)
            adj_matrix = adj_matrix.todense()
            adj_matrix = np.asarray(adj_matrix)
                
            if self.model_type == "FCN":
                network = network.flatten().reshape(1, -1)
                if self.MODE not in ["rank", "gcn"]:
                    if len(activated_idx):
                        network[0,activated_idx] = 0
                advbot = np.concatenate((history, network, activated), 1)
                obs = {
                    "action_mask": self.action_mask,
                    "avail_actions": self.action_assignments,
                    "advbot": advbot
                }
            else:
                obs = {
                    "action_mask": self.action_mask,
                    "avail_actions": self.action_assignments,
                    "advbot": network.reshape(network.shape[0], network.shape[1], 1),
                    "activated": activated,
                    "history": history,
                    "adjacency": adj_matrix
                }
        return obs


    def reset(self, reset_network=True):
        self.initialize(reset_network=reset_network)
        obs = self.pack_observation("level1")
        return {"level1": obs}


    def compute_influence(self, graph, seed_nodes, prob, n_iters=10):
        total_spead = 0
        for i in range(n_iters):
            np.random.seed(i)
            active = seed_nodes[:]
            new_active = seed_nodes[:]
            while new_active:
                activated_nodes = []
                for node in new_active:
                    neighbors = list(graph.neighbors(node))
                    success = np.random.uniform(0, 1, len(neighbors)) < prob
                    activated_nodes += list(np.extract(success, neighbors))
                new_active = list(set(activated_nodes) - set(active))
                active += new_active
            total_spead += len(active)
        return total_spead / n_iters


    def cal_rewards(self, test=False, seeds=None, action=None, specific_action=False, reward_shaping=1):
        if not seeds:
            seeds = list(self.seed_nodes.keys())

        if not test:
            if not specific_action:
                cur_reward = self.compute_influence(self.G, seeds, prob=self.PROB_RETWEET, n_iters=10)
                reward = cur_reward
            else:
                assert action >= 0
                cur_reward = self.compute_influence(self.G, [action], prob=self.PROB_RETWEET, n_iters=10)
                reward = cur_reward
                
        else:
            # print("SEEDS", seeds, len(seeds))
            print("out_degree", self.out_degree[seeds][:5], len(self.seed_nodes))
            cur_reward = self.compute_influence(self.G, seeds, prob=self.PROB_RETWEET, n_iters=10)
            reward = cur_reward

        if reward_shaping:
            reward = 1.0 * reward/self.best_reward()

        return reward


    def render(self, mode=None):
        pass



    def step(self, action_dict):
        assert len(action_dict) == 1, action_dict
        if "level1" in action_dict:
            return self._step_level1(action_dict["level1"])
        else:
            return self._step_level2(action_dict["level2"])


    def _step_level1(self, action):
        self.current_t += 1
        self.state += self.ACTION[action]
        detection_reward = 0.1
        
        if self.flg_detection:
            try:
                if len(self.state) > self.INTERVALS[self.current_interval]:
                # if self.current_t % self.INTERVAL == 0 and self.current_t > 0:
                    # print("CHECKING DETECTION", self.current_interval)
                    pred = self.detector.predict(self.state)[0]
                    if pred >= 0.5:
                        self.done = self.current_t
                    else:
                        detection_reward += (self.current_t - self.last_undetect)/self.MAX_TIME_STEP
                        self.last_undetect = self.current_t

                    self.current_interval += 1
            except:
                print(self.INTERVALS, len(self.INTERVALS))
                print(self.current_interval)

        if not self.validation:
            if len(self.state) >= self.MAX_TIME_STEP:
            # if self.current_t >= self.MAX_TIME_STEP:
                self.done = self.current_t
        else:
            if len(self.seed_nodes) >= self.MAX_TIME_STEP:
                self.done = self.current_t
            elif (self.current_t > self.UPPER_TIME_LIMIT):
                self.done = self.MAX_TIME_STEP

        self.level1_reward = detection_reward

        if self.ACTION[action] == "T":
            global_obs1 = self.pack_observation(agent="level1")
            if not self.done:
                return {"level1": global_obs1}, \
                        {"level1": 0.1 * self.level1_reward}, \
                        {"__all__":False}, \
                        {"level1": {}} 

            else:
                influence_reward = self.cal_rewards(specific_action=False, reward_shaping=self.REWARD_SHAPING)
                global_obs2 = self.pack_observation(agent="level2")
                if "R" in self.state or "A" in self.state or "M" in self.state:
                    return {"level1": global_obs1, "level2": global_obs2}, \
                            {"level1": influence_reward, "level2": influence_reward}, \
                            {"__all__":self.done}, \
                            {"level1": {}, "level2": {}} 
                else:
                    return {"level1": global_obs1}, \
                            {"level1": influence_reward}, \
                            {"__all__":self.done}, \
                            {"level1": {}} 

        else:
            global_obs2 = self.pack_observation(agent="level2")
            return {"level2": global_obs2}, \
                    {"level2": self.level2_reward}, \
                    {"__all__":False}, \
                    {"level2": {}}       


    def _step_level2(self, action):
        if self.validation:
            if len(self.seed_nodes) >= self.MAX_TIME_STEP:
                print("DONE HERE", len(self.seed_nodes), self.MAX_TIME_STEP)
                self.done = self.current_t
        
        if not (self.validation and self.done):
            self.seed_nodes[action] = 1

        self.following[action] = 1
        bot_degree = len(self.following)
        target_degree = self.G.degree(action)
        ratio = (1+bot_degree)/(1+target_degree)
        ratio = np.clip(ratio, a_min=0.0, a_max=1.0)

        if np.random.binomial(1, p=1-ratio):
            self.state += self.state[-1]
            self.state += self.state[-1]
            self.state += self.state[-1]

        # if not self.validation:
        self.update_avail_actions()
        
        global_obs1 = self.pack_observation(agent="level1")

        if not self.done:
            influence_reward = self.cal_rewards(action=action, specific_action=True, reward_shaping=self.REWARD_SHAPING)
            self.level2_reward = influence_reward
            
            return {"level1": global_obs1}, \
                {"level1": influence_reward}, \
                {"__all__": False}, \
                {"level1": {}}
        else:
            influence_reward = self.cal_rewards(specific_action=False, reward_shaping=self.REWARD_SHAPING)
            global_obs2 = self.pack_observation(agent="level2")
            return {"level1": global_obs1, "level2": global_obs2}, \
                {"level1": influence_reward, "level2": influence_reward}, \
                {"__all__": self.done}, \
                {"level1": {}, "level2": {}} 

    def close(self):
        self.reset()
