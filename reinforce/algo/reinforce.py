# Copyright (c) 2020 NVIDIA Corporation. All rights reserved.
# This work is licensed under a NVIDIA Open Source Non-commercial license.

import torch
import torch.nn as nn
import torch.optim as optim
from pdb import set_trace as bp


class REINFORCE():
    def __init__(self,
                 actor_critic,
                 entropy_coef,
                 lr=None,
                 eps=None,
                 alpha=None,
                 max_grad_norm=None,
                 acktr=False):

        self.actor_critic = actor_critic
        self.acktr = acktr
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        # self.optimizer = optim.Adam(actor_critic.parameters(), lr)#, eps=eps)
        self.optimizer = optim.SGD(actor_critic.parameters(), lr, momentum=0.9)#, eps=eps)

    def update(self, rollouts):
        obs_shape = rollouts.obs.size()[2:]
        action_shape = rollouts.actions.size()[-1]
        num_steps, num_processes, _ = rollouts.rewards.size()

        values, action_log_probs, dist_entropy, _, distribution = self.actor_critic.evaluate_actions(
            # rollouts.obs[:-1].view(-1, *obs_shape),
            # rollouts.recurrent_hidden_states[0].view(-1, self.actor_critic.recurrent_hidden_state_size),
            # rollouts.actions.view(-1, action_shape)
            rollouts.obs[:-1],
            rollouts.recurrent_hidden_states[0],
            rollouts.actions
        )

        values = values.view(num_steps, num_processes, 1)
        action_log_probs = action_log_probs.view(num_steps, num_processes, 1)

        # advantages = rollouts.returns[:-1] - values
        advantages = rollouts.returns[:-1]

        action_loss = -(advantages.detach() * action_log_probs).mean()

        self.optimizer.zero_grad()
        # (action_loss - dist_entropy * self.entropy_coef).backward()
        action_loss.backward()

        # nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.max_grad_norm)

        self.optimizer.step()

        return 0, action_loss.item(), dist_entropy.item()
