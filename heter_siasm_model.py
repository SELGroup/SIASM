import numpy as np
import torch
from torch import nn

from gym.spaces import Box

from ray.rllib.agents.dqn.dqn_torch_model import DQNTorchModel
from ray.rllib.models.torch.fcnet import FullyConnectedNetwork as TorchFC
from ray.rllib.models.torch.misc import AppendBiasLayer
from ray.rllib.models.torch.misc import SlimFC
from ray.rllib.models.torch.misc import normc_initializer
from ray.rllib.models.torch.visionnet import VisionNetwork as TorchConv
from ray.rllib.utils.torch_utils import FLOAT_MAX
from ray.rllib.utils.torch_utils import FLOAT_MIN
# from torch_geometric.nn import GraphConv
# from torch_geometric.data import Data
from sip import *
import networkx as nx

class GraphConvLayer(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super(GraphConvLayer, self).__init__()
        self.linear = torch.nn.Linear(input_dim, output_dim)

    def forward(self, node_features, adj_matrix):
        # Compute the adjacency matrix normalization
        rowsum = torch.sum(adj_matrix, dim=1, keepdim=True)
        normalized_adj_matrix = adj_matrix / (rowsum + 1e-8)

        # Perform graph convolution
        node_features = self.linear(node_features)
        output = torch.matmul(normalized_adj_matrix, node_features)

        return output

class RGCNLayer(nn.Module):
    def __init__(self, in_features, out_features, num_relations):
        super(RGCNLayer, self).__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.relation_weights = nn.ModuleList([nn.Linear(in_features, out_features) for _ in range(num_relations)])
        
    def forward(self, adjacency_list, node_features):
        # Perform relational graph convolution operation
        x = self.linear(node_features)
        for node in adjacency_list.keys():
            adj_nodes = adjacency_list[node]
            relation_weight = self.relation_weights[node]
            neighbor_features = torch.stack([node_features[n] for n in adj_nodes])
            print(neighbor_features.shape, relation_weight.weight.t().shape)
            x += torch.matmul(neighbor_features, relation_weight.weight.t())
        x = torch.relu(x)
        return x

class RGCNModel(nn.Module):
    def __init__(self, num_node_features, num_relations):
        super(RGCNModel, self).__init__()
        self.conv1 = RGCNLayer(num_node_features, 64, num_relations)
        self.conv2 = RGCNLayer(64, 32, num_relations)
        self.fc = nn.Linear(32, num_node_features)
        
    def forward(self, adjacency_list, node_features):
        x = self.conv1(adjacency_list, node_features)
        x = self.conv2(adjacency_list, x)
        x = torch.mean(x, dim=0)  # Aggregate node features
        x = self.fc(x)
        return x

class TorchParametricActionsModel(DQNTorchModel):
    def __init__(self,
                 obs_space,
                 action_space,
                 num_outputs,
                 model_config,
                 name,
                 node_embed_dim=2,
                 true_obs_shape=(None, ),
                 activated_obs_shape=(None, ),
                 history_obs_shape=(None, ),
                 action_embed_size=None,
                 model_type="FCN",
                 num_filters=4,
                 **kw):
        DQNTorchModel.__init__(self, obs_space, action_space, num_outputs,
                               model_config, name, **kw)
        self.model_type = model_type
        self.vf_share_layers = True
        self.iteration = 0

        if model_type == "FCN":
            self.action_embed_model = TorchFC(
            Box(-1, 1, shape=true_obs_shape), action_space, action_embed_size,
            model_config, name + "_action_embed")

        else:
            self.action_embed_model = TorchConv(
            Box(-1, 1, shape=true_obs_shape), action_space, int(action_embed_size/3),
            {"conv_filters": [[num_filters, [node_embed_dim,node_embed_dim], true_obs_shape[0]-1]]}, name + "_action_embed")

            self.activated_embed_model = TorchFC(
            Box(-1, 1, shape=activated_obs_shape), action_space, int(action_embed_size/3),
            model_config, name + "_activated_embed")

            self.history_embed_model = TorchFC(
            Box(-1, 1, shape=history_obs_shape), action_space, int(action_embed_size/3),
            model_config, name + "_history_embed")

            self.rgcn = RGCNModel(action_embed_size, 4)
            self.gcn = GraphConvLayer(node_embed_dim, node_embed_dim)


    def forward(self, input_dict, state, seq_lens):
        self.iteration += 1
        avail_actions = input_dict["obs"]["avail_actions"]
        action_mask = input_dict["obs"]["action_mask"]
        # action embedding based on RGCN
        adj_matrix = torch.FloatTensor(input_dict["obs"]["adjacency"].cpu())
        # adj_list = {}
        # for i in range(adj_matrix.shape[1]):
        #     adj_i = []
        #     length_i = 0
        #     for j in range(adj_matrix.shape[0]):
        #         adj_j = []
        #         for k in range(adj_matrix.shape[2]):
        #             if adj_matrix[j][i][k] > 0.0:
        #                 adj_j.append(k)
        #         adj_i.append(adj_j)
        #         length_i += len(adj_j)
        #     if length_i > 0:
        #         adj_list[i] = np.array(adj_i)
        # adj_matrix = torch.FloatTensor(input_dict["obs"]["adjacency"])
        action_embed = self.gcn(input_dict["obs"]["advbot"].squeeze(), adj_matrix)
        # if len(adj_list) == 0:
        #     adj_matrix = torch.FloatTensor(input_dict["obs"]["adjacency"])
        #     action_embed = self.gcn(input_dict["obs"]["advbot"].squeeze(), adj_matrix)
        # else:
        #     action_embed = self.rgcn(adj_list, input_dict["obs"]["advbot"].squeeze())
        # end embedding
        action_embed_new, _ = self.action_embed_model({"obs": action_embed.unsqueeze(axis=-1)})
        # user filter based on structural information principles
        for i in range(action_mask.shape[0]):
            if (self.iteration % 15000) == 0:
                disable_node = self.update_avail_actions(input_dict["obs"]["adjacency"][i], action_embed[i])
                action_mask[i][disable_node] = 0
        # end filter
        activated_embed, _ = self.activated_embed_model({"obs": input_dict["obs"]["activated"]})
        history_embed, _ = self.history_embed_model({"obs": input_dict["obs"]["history"]})
        features = torch.cat((action_embed_new, activated_embed, history_embed), 1)

        intent_vector = torch.unsqueeze(features, 1) #32, 1, 50
        action_logits = torch.sum(avail_actions * intent_vector, dim=2)
        inf_mask = torch.clamp(torch.log(action_mask), FLOAT_MIN, FLOAT_MAX)

        return action_logits + inf_mask, state

    def value_function(self):
        return self.action_embed_model.value_function() + \
        self.activated_embed_model.value_function() + \
        self.history_embed_model.value_function()

    def update_avail_actions(self, old_adj_matrix, action_embeds):
        adj_matrix = np.zeros([old_adj_matrix.shape[0], old_adj_matrix.shape[0]], dtype=np.float32)
        for v1 in range(old_adj_matrix.shape[0]):
            for v2 in range(old_adj_matrix.shape[1]):
                if old_adj_matrix[v1][v2] > 0:
                    weight = abs(np.corrcoef(action_embeds[v1].detach().numpy(), action_embeds[v2].detach().numpy())[0, 1])
                    if not math.isnan(weight):
                        adj_matrix[v1][v2] += weight
                        adj_matrix[v2][v1] += weight
        pt = PartitionTree(adj_matrix=adj_matrix)
        pt.build_coding_tree(3)  # tree height
        ens = []
        ids = []
        for id in pt.tree_node.keys():
            node = pt.tree_node[id]
            if node.child_h == 1:
                ids.append(id)
                ens.append(pt.community_entropy(pt.tree_node, id, 0.0))
        disable_node = []
        idx = np.argsort(ens)
        vns = 0
        for i in range(len(idx)):
            id = ids[idx[i]]
            for vid in pt.tree_node[id].partition:
                disable_node.append(vid)
            vns += len(pt.tree_node[id].partition)
            if vns >= 0.01 * old_adj_matrix.shape[0]:
                break
        return disable_node
