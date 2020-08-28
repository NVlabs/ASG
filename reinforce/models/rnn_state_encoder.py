# Copyright (c) 2020 NVIDIA Corporation. All rights reserved.
# This work is licensed under a NVIDIA Open Source Non-commercial license.

import torch
import torch.nn as nn
from pdb import set_trace as bp


class RNNStateEncoder(nn.Module):
    def __init__(self, input_size: int = 1, hidden_size: int = 1, num_layers: int = 1, rnn_type: str = "LSTM"):
        r"""An RNN for encoding the state in RL.

        Supports masking the hidden state during various timesteps in the forward lass

        Args:
            input_size: The input size of the RNN
            hidden_size: The hidden size
            num_layers: The number of recurrent layers
            rnn_type: The RNN cell type.  Must be GRU or LSTM
        """

        super().__init__()
        self._num_recurrent_layers = num_layers
        self._rnn_type = rnn_type

        self.rnn = getattr(nn, rnn_type)(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
        )

        self.layer_init()

    def layer_init(self):
        for name, param in self.rnn.named_parameters():
            if "weight" in name:
                nn.init.orthogonal_(param)
            elif "bias" in name:
                nn.init.constant_(param, 0)

    @property
    def num_recurrent_layers(self):
        return self._num_recurrent_layers * (
            2 if "LSTM" in self._rnn_type else 1
        )

    def _pack_hidden(self, hidden_states):
        if "LSTM" in self._rnn_type:
            hidden_states = torch.cat(
                [hidden_states[0], hidden_states[1]], dim=0
            )
        return hidden_states

    def _unpack_hidden(self, hidden_states):
        if "LSTM" in self._rnn_type:
            hidden_states = (
                hidden_states[0 : self._num_recurrent_layers],
                hidden_states[self._num_recurrent_layers :],
            )
        return hidden_states

    def single_forward(self, x, hidden_states):
        r"""Forward for a non-sequence input
        """
        if len(x.size()) == 2:
            x = x.unsqueeze(0)
        # input: (seq_len, batch, input_size)
        x, hidden_states = self.rnn(x, hidden_states)
        return x, hidden_states

    def forward(self, x, hidden_states):
        hidden_states = self._unpack_hidden(hidden_states)
        x, hidden_states = self.single_forward(x, hidden_states)
        hidden_states = self._pack_hidden(hidden_states)
        return x, hidden_states
