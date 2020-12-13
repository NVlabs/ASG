# Copyright (c) 2020 NVIDIA Corporation. All rights reserved.
# This work is licensed under a NVIDIA Open Source Non-commercial license.

import abc
from pdb import set_trace as bp
import numpy as np
import torch
import torch.nn as nn

from reinforce.distributions import Categorical
from reinforce.models.rnn_state_encoder import RNNStateEncoder


class Policy(nn.Module):
    def __init__(self, coord_size, input_size=(1, 1), action_space=1, hidden_size=1, window_size=1):
        # input_size: (#lstm_input, #mlp_input)
        super().__init__()
        self.net = BasicNet(coord_size, input_size=input_size, hidden_size=hidden_size, window_size=window_size)
        # will coordinate-wisely return distributions
        self.action_distribution = Categorical(input_size[0]*hidden_size+input_size[1]+1, action_space, coord_size=coord_size)
        self.critic = CriticHead(coord_size * (input_size[0]*hidden_size+input_size[1]+1))
        self.recurrent_hidden_state_size = hidden_size
        self.coord_size = coord_size
        self.input_size = input_size
        self.action_space = action_space
        self.hidden_size = hidden_size
        self.window_size = window_size

    def forward(self, *x):
        raise NotImplementedError

    def act(self, observations, rnn_hidden_states, deterministic=False):
        features, rnn_hidden_states = self.net(observations, rnn_hidden_states)
        distribution = self.action_distribution(features)
        # (coord, seq_len*batch, feature) ==> (seq_len*batch, coord, feature)
        value = self.critic(features.permute(1, 0, 2).view(features.size(1), -1))

        if deterministic:
            action = distribution.mode()
        else:
            action = distribution.sample()

        action_log_probs = distribution.log_probs(action)
        return value, action, action_log_probs, rnn_hidden_states, distribution

    def get_value(self, observations, rnn_hidden_states):
        features, _ = self.net(observations, rnn_hidden_states)
        # features = features.view(-1, self.batch_size * self.recurrent_hidden_state_size)
        return self.critic(features.permute(1, 0, 2).view(features.size(1), -1))

    def evaluate_actions(self, observations, rnn_hidden_states, action):
        features, rnn_hidden_states = self.net(observations, rnn_hidden_states)
        # features = features.view(-1, self.batch_size * self.recurrent_hidden_state_size)
        distribution = self.action_distribution(features)
        value = self.critic(features.permute(1, 0, 2).contiguous().view(features.size(1), -1))

        action_log_probs = distribution.log_probs(action)
        distribution_entropy = distribution.entropy().mean()

        return value, action_log_probs, distribution_entropy, rnn_hidden_states, distribution



class CriticHead(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.fc = nn.Linear(input_size, 1)
        nn.init.orthogonal_(self.fc.weight)
        nn.init.constant_(self.fc.bias, 0)

    def forward(self, x):
        return self.fc(x)


class Net(nn.Module, metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def forward(self, observations, rnn_hidden_states, prev_actions):
        pass

    @property
    @abc.abstractmethod
    def output_size(self):
        pass

    @property
    @abc.abstractmethod
    def num_recurrent_layers(self):
        pass


class BasicNet(Net):
    def __init__(self, coord_size, input_size=(1, 1), hidden_size=1, window_size=1):
        super().__init__()
        self._coord_size = coord_size
        # input_size: (#lstm_input, #mlp_input)
        self._input_size = input_size
        self._hidden_size = hidden_size
        self._window_size = window_size
        self.state_encoder = nn.ModuleList([
            RNNStateEncoder(input_size=window_size, hidden_size=self._hidden_size)
            for _ in range(input_size[0])
        ])
        self.train()

    @property
    def output_size(self):
        return self._hidden_size

    @property
    def num_recurrent_layers(self):
        return self.state_encoder[0].num_recurrent_layers

    def forward(self, observations, rnn_hidden_states):
        # observation: (seq_len, batch_size, #lstm_input * window + #scalar_input + #actions * 1(LR))
        # rnn_hidden_states: (#lstm_input * hidden_size)
        outputs = []
        rnn_hidden_states_new = []
        # coordinate-wise
        for i in range(self._input_size[0]):
            # output: (seq_len, batch(1), hidden_size)
            output, rnn_hidden_state = self.state_encoder[i](observations[:, :, i*self._window_size:(i+1)*self._window_size], rnn_hidden_states[:, :, i*self._hidden_size:(i+1)*self._hidden_size])
            outputs.append(output)
            rnn_hidden_states_new.append(rnn_hidden_state)
        # outputs: (seq_len, batch(1), hidden_size * #lstm_input + #scalar_input)
        outputs = torch.cat(outputs + [observations[:, :, self._input_size[0]*self._window_size:self._input_size[0]*self._window_size+self._input_size[1]]], dim=2)
        # add LR feature for each coord
        outputs_LR = []
        for coord in range(-self._coord_size, 0):
            outputs_LR.append(torch.cat([outputs, observations[:, :, observations.size(2)+coord:observations.size(2)+coord+1]], dim=2))
        outputs_LR = torch.stack(outputs_LR, dim=0) # (coord, seq_len, 1, hidden_size * #lstm_input + #scalar_input + 1)
        outputs_LR = outputs_LR.view(self._coord_size, -1, outputs_LR.size(-1)) # (coord, seq_len * 1, hidden_size * #lstm_input + #scalar_input + 1)
        return outputs_LR, rnn_hidden_states
