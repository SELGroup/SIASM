import numpy as np
import torch

from gym.spaces import Box

from ray.rllib.agents.dqn.dqn_torch_model import DQNTorchModel
from ray.rllib.models.torch.fcnet import FullyConnectedNetwork as TorchFC
from ray.rllib.models.torch.misc import AppendBiasLayer
from ray.rllib.models.torch.misc import SlimFC
from ray.rllib.models.torch.misc import normc_initializer
from ray.rllib.models.torch.visionnet import VisionNetwork as TorchConv
from ray.rllib.utils.torch_ops import FLOAT_MAX
from ray.rllib.utils.torch_ops import FLOAT_MIN

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


    def forward(self, input_dict, state, seq_lens):
        avail_actions = input_dict["obs"]["avail_actions"]
        action_mask = input_dict["obs"]["action_mask"]
        action_embed, _ = self.action_embed_model({"obs": input_dict["obs"]["advbot"]})
        activated_embed, _ = self.activated_embed_model({"obs": input_dict["obs"]["activated"]})
        history_embed, _ = self.history_embed_model({"obs": input_dict["obs"]["history"]})
        features = torch.cat((action_embed, activated_embed, history_embed), 1)

        intent_vector = torch.unsqueeze(features, 1) #32, 1, 50
        action_logits = torch.sum(avail_actions * intent_vector, dim=2)
        inf_mask = torch.clamp(torch.log(action_mask), FLOAT_MIN, FLOAT_MAX)

        return action_logits + inf_mask, state

    def value_function(self):
        return self.action_embed_model.value_function() + \
        self.activated_embed_model.value_function() + \
        self.history_embed_model.value_function()
