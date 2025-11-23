# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

import torch
import torch.nn as nn
import torch.optim as optim

import torch.nn.functional as F


from rsl_rl.modules import ActorCritic_DWAQ
from rsl_rl.storage import RolloutStorage
import wandb
from rsl_rl.utils import unpad_trajectories


class RMS(object):
    def __init__(self, device, epsilon=1e-4, shape=(1,)):
        self.M = torch.zeros(shape, device=device)
        self.S = torch.ones(shape, device=device)
        self.n = epsilon

    def __call__(self, x):
        bs = x.size(0)
        delta = torch.mean(x, dim=0) - self.M
        new_M = self.M + delta * bs / (self.n + bs)
        new_S = (self.S * self.n + torch.var(x, dim=0) * bs + (delta**2) * self.n * bs / (self.n + bs)) / (self.n + bs)

        self.M = new_M
        self.S = new_S
        self.n += bs

        return self.M, self.S


class Estimator(nn.Module):
    def __init__(self,  input_dim,
                        output_dim,
                        hidden_dims=[128, 64],
                        activation="elu",
                        **kwargs):
        super(Estimator, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        activation = nn.ELU()
        estimator_layers = []
        estimator_layers.append(nn.Linear(self.input_dim, hidden_dims[0]))
        estimator_layers.append(activation)
        for l in range(len(hidden_dims)):
            if l == len(hidden_dims) - 1:
                estimator_layers.append(nn.Linear(hidden_dims[l], output_dim))
            else:
                estimator_layers.append(nn.Linear(hidden_dims[l], hidden_dims[l + 1]))
                estimator_layers.append(activation)
        # estimator_layers.append(nn.Tanh())
        self.estimator = nn.Sequential(*estimator_layers)
    
    def forward(self, input):
        return self.estimator(input)
    
    def inference(self, input):
        with torch.no_grad():
            return self.estimator(input)
        

class PPO:
    actor_critic: ActorCritic_DWAQ
    def __init__(self,
                 actor_critic,
                #  estimator,
                #  estimator_paras,
                 depth_encoder,
                 depth_encoder_paras,
                 depth_actor,
                 num_learning_epochs=1,
                 num_mini_batches=1,
                 clip_param=0.2,
                 gamma=0.998,
                 lam=0.95,
                 value_loss_coef=1.0,
                 entropy_coef=0.0,
                 learning_rate=1e-3,
                 max_grad_norm=1.0,
                 use_clipped_value_loss=True,
                 schedule="fixed",
                 desired_kl=0.01,
                 device='cpu',
                 dagger_update_freq=20,
                 priv_reg_coef_schedual = [0, 0, 0],              # [0, 0.1, 2000, 3000]
                 **kwargs
                 ):

        
        self.device = device

        self.desired_kl = desired_kl
        self.schedule = schedule
        self.learning_rate = learning_rate

        # PPO components
        self.actor_critic = actor_critic
        self.actor_critic.to(self.device)
        self.storage = None # initialized later
        self.optimizer = optim.Adam(self.actor_critic.parameters(), lr=learning_rate)
        self.transition = RolloutStorage.Transition()

        # PPO parameters
        self.clip_param = clip_param
        self.num_learning_epochs = num_learning_epochs
        self.num_mini_batches = num_mini_batches
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.gamma = gamma
        self.lam = lam
        self.max_grad_norm = max_grad_norm
        self.use_clipped_value_loss = use_clipped_value_loss

        # Adaptation
        self.hist_encoder_optimizer = optim.Adam(self.actor_critic.history_encoder.parameters(), lr=learning_rate)

        self.priv_reg_coef_schedual = priv_reg_coef_schedual
        self.counter = 0


        self.estimator = Estimator(53, 3, hidden_dims=[128, 64], activation="elu").to(self.device)
        self.estimator_optimizer = optim.Adam(self.estimator.parameters(), lr=1e-3)
    

        # if self.if_depth:
        # self.depth_actor_optimizer = optim.Adam(self.actor_critic.parameters(), lr=1.e-3)

        # Depth encoder
        self.if_depth = depth_encoder != None
        if self.if_depth:
            self.depth_encoder = depth_encoder
            self.depth_encoder_paras = depth_encoder_paras
            self.depth_actor = depth_actor
            self.depth_actor_optimizer = optim.Adam([*self.depth_actor.parameters(), *self.depth_encoder.parameters()], lr=depth_encoder_paras["learning_rate"])

        # Estimator
        # self.estimator = estimator
        # self.priv_states_dim = estimator_paras["priv_states_dim"]     # 9
        # self.num_prop = estimator_paras["num_prop"]                   # 53
        # self.num_scan = estimator_paras["num_scan"]                   # 132
        # self.estimator_optimizer = optim.Adam(self.estimator.parameters(), lr=estimator_paras["learning_rate"])
        # self.train_with_estimated_states = estimator_paras["train_with_estimated_states"]

        # # Depth encoder
        # self.if_depth = depth_encoder != None
        # if self.if_depth:
        #     self.depth_encoder = depth_encoder
        #     self.depth_encoder_optimizer = optim.Adam(self.depth_encoder.parameters(), lr=depth_encoder_paras["learning_rate"])
        #     self.depth_encoder_paras = depth_encoder_paras
        #     self.depth_actor = depth_actor
        #     self.depth_actor_optimizer = optim.Adam([*self.depth_actor.parameters(), *self.depth_encoder.parameters()], lr=depth_encoder_paras["learning_rate"])

    def init_storage(self, num_envs, num_transitions_per_env, actor_obs_shape, critic_obs_shape, action_shape):
        self.storage = RolloutStorage(num_envs, num_transitions_per_env, actor_obs_shape,  critic_obs_shape, action_shape, self.device)

    def test_mode(self):
        self.actor_critic.test()
    
    def train_mode(self):
        self.actor_critic.train()

    def act(self, obs, critic_obs, info, hist_encoding=False, gen_data = False):
        # if self.actor_critic.is_recurrent:               # False
        #     self.transition.hidden_states = self.actor_critic.get_hidden_states()
        # # Compute the actions and values, use proprio to compute estimated priv_states then actions, but store true priv_states
        # if self.train_with_estimated_states:
        #     obs_est = obs.clone()                                                       #  size 753
        #     priv_states_estimated = self.estimator(obs_est[:, :self.num_prop])          # output dimension is 9, this is to estimate privi_explicit term. it is defined in on_policy_runner.py: Estimator
        #     obs_est[:, self.num_prop+self.num_scan:self.num_prop+self.num_scan+self.priv_states_dim] = priv_states_estimated  # self.priv_states_dim is 9.  obs_est dimension is 753
        #     self.transition.actions = self.actor_critic.act(obs_est, hist_encoding).detach()   # It is using def act() function in Line 302 of actor_critic.py
        # else:
        #     self.transition.actions = self.actor_critic.act(obs, hist_encoding).detach()


        obs_est = obs.clone()
        priv_states_estimated = self.estimator(obs_est[:, :53])
        obs_est[:, 53+132:53+132+3] = priv_states_estimated


        actions, _ = self.actor_critic.act(obs_est, info["depth"], hist_encoding, gen_data=gen_data)
        self.transition.actions = actions.detach()
        
        self.transition.values = self.actor_critic.evaluate(critic_obs).detach()   # [num_envs, 1]
        self.transition.actions_log_prob = self.actor_critic.get_actions_log_prob(self.transition.actions).detach()
        self.transition.action_mean = self.actor_critic.action_mean.detach()
        self.transition.action_sigma = self.actor_critic.action_std.detach()
        self.transition.observations = obs                                         # [num_envs, 753]
        # self.transition.prev_obs = prev_obs
        self.transition.depth_extras = info["depth"]

        
        self.transition.critic_observations = critic_obs


        # # for depth image
        # if "depth" in info and info["depth"] is not None and global_counter % update_interval == 0:
        #     self.transition.depth_extras = info["depth"]  # [num_envs, 58, 87]
        # else:
        #     self.transition.depth_extras = None


        return self.transition.actions
    
    def process_env_step(self, rewards, dones, infos):
        rewards_total = rewards.clone()

        self.transition.rewards = rewards_total.clone()
        self.transition.dones = dones
        # Bootstrapping on time outs
        if 'time_outs' in infos:
            self.transition.rewards += self.gamma * torch.squeeze(self.transition.values * infos['time_outs'].unsqueeze(1).to(self.device), 1)

        # Record the transition
        self.storage.add_transitions(self.transition)
        self.transition.clear()
        self.actor_critic.reset(dones)

        return rewards_total
    
    def compute_returns(self, last_critic_obs):
        last_values= self.actor_critic.evaluate(last_critic_obs).detach()
        self.storage.compute_returns(last_values, self.gamma, self.lam)
    

    def update(self, gen_data = False):
        mean_value_loss = 0
        mean_surrogate_loss = 0
        mean_estimator_loss = 0
        mean_discriminator_loss = 0
        mean_discriminator_acc = 0
        mean_priv_reg_loss = 0


        mean_autoenc_loss = 0
        mean_vel_target_loss = 0
        # mean_hf_target_loss = 0
        mean_obs_prop_target_loss = 0
        mean_obs_scan_target_loss = 0
        mean_priv_reg_loss = 0


        # add image_batch
        generator = self.storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
        for obs_batch, critic_obs_batch, actions_batch, target_values_batch, advantages_batch, returns_batch, old_actions_log_prob_batch, \
            old_mu_batch, old_sigma_batch, hid_states_batch, masks_batch, image_batch in generator:


                # --- Estimator step (with grads) ---
                est_in = obs_batch[:, :53].detach()
                priv_states_estimated = self.estimator(est_in)                 # requires_grad=True
                estimator_loss = (priv_states_estimated - obs_batch[:, 53+132:53+132+3].detach()).pow(2).mean()

                self.estimator_optimizer.zero_grad()
                estimator_loss.backward()
                nn.utils.clip_grad_norm_(self.estimator.parameters(), self.max_grad_norm)
                self.estimator_optimizer.step()

                # --- Recompute or detach before PPO step ---
                # Option A (recommended): recompute under no_grad so it matches the updated estimator
                with torch.no_grad():
                    priv_est_for_ppo = self.estimator(est_in)

                # Option B: use the first output but detach it
                # priv_est_for_ppo = priv_states_estimated.detach()

                # Inject into obs for the actor_critic
                obs_est_batch = obs_batch.detach().clone()
                obs_est_batch[:, 53+132:53+132+3] = priv_est_for_ppo

                _, decoded_scan = self.actor_critic.act(obs_est_batch, image_batch, hist_encoding=False, masks=masks_batch, hidden_states=hid_states_batch[0], gen_data = False) # match distribution dimension. It is using def act() function in Line 302 of actor_critic.py
                
                # added scan reconstruction loss
                obs_scan_target = obs_batch[:, 53:185].detach()
                # obs_scan_target_loss = nn.MSELoss()(decoded_scan, obs_scan_target)
                # autoenc_loss = (obs_scan_target_loss) / self.num_mini_batches
                obs_scan_target_loss = F.mse_loss(decoded_scan, obs_scan_target, reduction='mean')
                autoenc_loss = obs_scan_target_loss / self.num_mini_batches
                
                
                actions_log_prob_batch = self.actor_critic.get_actions_log_prob(actions_batch)
                value_batch = self.actor_critic.evaluate(critic_obs_batch, masks=masks_batch, hidden_states=hid_states_batch[1])
                mu_batch = self.actor_critic.action_mean
                sigma_batch = self.actor_critic.action_std
                entropy_batch = self.actor_critic.entropy

                
                # Adaptation module update
                priv_batch = obs_batch[:, 194:223].detach()   # 29
                priv_latent_batch = self.actor_critic.infer_priv_latent(priv_batch)       # using privilege latent and priv_encoder (input is 29, output is 20) directly, output is 20
                with torch.inference_mode():
                    history_batch = obs_batch[:, -530:]
                    hist_latent_batch = self.actor_critic.infer_hist_latent(history_batch)   # infer privilege latent using history data, output is 20
                
                priv_reg_loss = (priv_latent_batch - hist_latent_batch.detach()).norm(p=2, dim=1).mean()
                priv_reg_stage = min(max((self.counter - self.priv_reg_coef_schedual[2]), 0) / self.priv_reg_coef_schedual[3], 1)   # self.counter +=1 for each update # priv_reg_coef_schedual = [0, 0.1, 2000, 3000]
                priv_reg_coef = priv_reg_stage * (self.priv_reg_coef_schedual[1] - self.priv_reg_coef_schedual[0]) + self.priv_reg_coef_schedual[0]

                ###################################################################################################################################
                
                # KL Divergence
                if self.desired_kl != None and self.schedule == 'adaptive':                #True
                    with torch.inference_mode():
                        kl = torch.sum(
                            torch.log(sigma_batch / old_sigma_batch + 1.e-5) + (torch.square(old_sigma_batch) + torch.square(old_mu_batch - mu_batch)) / (2.0 * torch.square(sigma_batch)) - 0.5, axis=-1)
                        kl_mean = torch.mean(kl)

                        if kl_mean > self.desired_kl * 2.0:
                            self.learning_rate = max(1e-5, self.learning_rate / 1.5)
                        elif kl_mean < self.desired_kl / 2.0 and kl_mean > 0.0:
                            self.learning_rate = min(1e-2, self.learning_rate * 1.5)
                        
                        for param_group in self.optimizer.param_groups:
                            param_group['lr'] = self.learning_rate


                # Surrogate loss           the objective function includes a clipped term that limits the change in policy to indirectly control the KL divergence
                ratio = torch.exp(actions_log_prob_batch - torch.squeeze(old_actions_log_prob_batch))   # actions_log_prob_batch size is 36864
                surrogate = -torch.squeeze(advantages_batch) * ratio
                surrogate_clipped = -torch.squeeze(advantages_batch) * torch.clamp(ratio, 1.0 - self.clip_param,
                                                                                1.0 + self.clip_param)
                surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()   # The clipping mechanism prevents the policy from changing too drastically


                # Value function loss
                if self.use_clipped_value_loss:                         # True         
                    value_clipped = target_values_batch + (value_batch - target_values_batch).clamp(-self.clip_param,
                                                                                                    self.clip_param)
                    value_losses = (value_batch - returns_batch).pow(2)             # value_batch is the current value function, target_values_batch is the old value function
                    value_losses_clipped = (value_clipped - returns_batch).pow(2)   # returns_batch is defined in Line 124 of rollout_storage.py, it is target values function
                    value_loss = torch.max(value_losses, value_losses_clipped).mean()     # self.returns[step] = advantage + self.values[step]
                else:
                    value_loss = (returns_batch - value_batch).pow(2).mean()

                loss = surrogate_loss + \
                       self.value_loss_coef * value_loss - \
                       self.entropy_coef * entropy_batch.mean() + \
                       autoenc_loss + \
                       priv_reg_coef * priv_reg_loss



                # Gradient step
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.max_grad_norm)
                self.optimizer.step()

                mean_value_loss += value_loss.item()
                mean_surrogate_loss += surrogate_loss.item()
                mean_estimator_loss += estimator_loss.item()
                # mean_priv_reg_loss += priv_reg_loss.item()
                mean_discriminator_loss += 0
                mean_discriminator_acc += 0



                mean_autoenc_loss += autoenc_loss.item()
                # mean_vel_target_loss += vel_target_loss.item()
                # # mean_hf_target_loss += hf_target_loss.item()
                # mean_obs_prop_target_loss += obs_prop_target_loss.item()
                mean_obs_scan_target_loss += obs_scan_target_loss.item()
                # mean_priv_reg_loss += priv_reg_loss.item()



        num_updates = self.num_learning_epochs * self.num_mini_batches
        mean_value_loss /= num_updates
        mean_surrogate_loss /= num_updates
        mean_estimator_loss /= num_updates
        mean_priv_reg_loss /= num_updates
        mean_discriminator_loss /= num_updates
        mean_discriminator_acc /= num_updates

        mean_autoenc_loss /= num_updates
        mean_vel_target_loss /= num_updates
        # mean_hf_target_loss /= num_updates
        mean_obs_prop_target_loss /= num_updates
        mean_obs_scan_target_loss /= num_updates
        mean_priv_reg_loss /= num_updates

        # reset CNN latent
        # self.actor_critic.reset_cnn_latent()

        self.storage.clear()
        self.update_counter()
        # return mean_value_loss, mean_surrogate_loss, mean_estimator_loss, mean_discriminator_loss, mean_discriminator_acc, mean_priv_reg_loss
        return mean_value_loss, mean_surrogate_loss, mean_estimator_loss, mean_discriminator_loss, mean_discriminator_acc, mean_autoenc_loss, mean_vel_target_loss, mean_obs_prop_target_loss, mean_obs_scan_target_loss, mean_priv_reg_loss

   
    def update_dagger(self):
        mean_hist_latent_loss = 0
        # if self.actor_critic.is_recurrent:
        #     generator = self.storage.reccurent_mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
        # else:
        #     generator = self.storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
        generator = self.storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
        for obs_batch, critic_obs_batch, actions_batch, target_values_batch, advantages_batch, returns_batch, old_actions_log_prob_batch, \
            old_mu_batch, old_sigma_batch, hid_states_batch, masks_batch, image_batch  in generator:
                with torch.inference_mode():
                    self.actor_critic.act(obs_batch, image_batch, hist_encoding=True, masks=masks_batch, hidden_states=hid_states_batch[0], gen_data=False) # It is using def act() function in Line 302 of actor_critic.py
                # Adaptation module update
                with torch.inference_mode():
                    priv_batch = obs_batch[:, 194:223]   # 29
                    priv_latent_batch = self.actor_critic.infer_priv_latent(priv_batch)
                history_batch = obs_batch[:, -530:]
                hist_latent_batch = self.actor_critic.infer_hist_latent(history_batch)

                hist_latent_loss = (priv_latent_batch.detach() - hist_latent_batch).norm(p=2, dim=1).mean()
                self.hist_encoder_optimizer.zero_grad()
                hist_latent_loss.backward()
                nn.utils.clip_grad_norm_(self.actor_critic.history_encoder.parameters(), self.max_grad_norm)
                self.hist_encoder_optimizer.step()
                
                mean_hist_latent_loss += hist_latent_loss.item()
        num_updates = self.num_learning_epochs * self.num_mini_batches
        mean_hist_latent_loss /= num_updates
        self.storage.clear()
        self.update_counter()
        return mean_hist_latent_loss
    
    
    # def update_depth_encoder(self, depth_latent_batch, scandots_latent_batch):           # not used
    #     # Depth encoder ditillation
    #     if self.if_depth:
    #         # TODO: needs to save hidden states
    #         depth_encoder_loss = (scandots_latent_batch.detach() - depth_latent_batch).norm(p=2, dim=1).mean()

    #         self.depth_encoder_optimizer.zero_grad()
    #         depth_encoder_loss.backward()
    #         nn.utils.clip_grad_norm_(self.depth_encoder.parameters(), self.max_grad_norm)
    #         self.depth_encoder_optimizer.step()
    #         return depth_encoder_loss.item()


    def update_depth_actor(self, actions_student_batch, actions_teacher_batch, yaw_student_batch, yaw_teacher_batch, depth_latent_student_batch, depth_latent_teacher_batch):   # only this one is used
        # if self.if_depth:

        depth_actor_loss = (actions_teacher_batch.detach() - actions_student_batch).norm(p=2, dim=1).mean()
        yaw_loss = (yaw_teacher_batch.detach() - yaw_student_batch).norm(p=2, dim=1).mean()
        depth_latent_loss = (depth_latent_teacher_batch.detach() - depth_latent_student_batch).norm(p=2, dim=1).mean()

        loss = depth_actor_loss + yaw_loss + depth_latent_loss

        self.depth_actor_optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.depth_actor.parameters(), self.max_grad_norm)
        self.depth_actor_optimizer.step()
        return depth_actor_loss.item(), yaw_loss.item(), depth_latent_loss.item()

    # def update_depth_actor(self, actions_student_batch, actions_teacher_batch, yaw_student_batch, yaw_teacher_batch):   # only this one is used
    #     if self.if_depth:
    #         depth_actor_loss = (actions_teacher_batch.detach() - actions_student_batch).norm(p=2, dim=1).mean()
    #         yaw_loss = (yaw_teacher_batch.detach() - yaw_student_batch).norm(p=2, dim=1).mean()

    #         loss = depth_actor_loss + yaw_loss

    #         self.depth_actor_optimizer.zero_grad()
    #         loss.backward()
    #         nn.utils.clip_grad_norm_(self.depth_actor.parameters(), self.max_grad_norm)
    #         self.depth_actor_optimizer.step()
    #         return depth_actor_loss.item(), yaw_loss.item()
    
    def update_depth_both(self, depth_latent_batch, scandots_latent_batch, actions_student_batch, actions_teacher_batch):          # not used
        if self.if_depth:
            depth_encoder_loss = (scandots_latent_batch.detach() - depth_latent_batch).norm(p=2, dim=1).mean()
            depth_actor_loss = (actions_teacher_batch.detach() - actions_student_batch).norm(p=2, dim=1).mean()

            depth_loss = depth_encoder_loss + depth_actor_loss

            self.depth_actor_optimizer.zero_grad()
            depth_loss.backward()
            nn.utils.clip_grad_norm_([*self.depth_actor.parameters(), *self.depth_encoder.parameters()], self.max_grad_norm)
            self.depth_actor_optimizer.step()
            return depth_encoder_loss.item(), depth_actor_loss.item()
    
    def update_counter(self):
        self.counter += 1
    
    def compute_apt_reward(self, source, target):

        b1, b2 = source.size(0), target.size(0)
        # (b1, 1, c) - (1, b2, c) -> (b1, 1, c) - (1, b2, c) -> (b1, b2, c) -> (b1, b2)
        # sim_matrix = torch.norm(source[:, None, ::2].view(b1, 1, -1) - target[None, :, ::2].view(1, b2, -1), dim=-1, p=2)
        # sim_matrix = torch.norm(source[:, None, :2].view(b1, 1, -1) - target[None, :, :2].view(1, b2, -1), dim=-1, p=2)
        sim_matrix = torch.norm(source[:, None, :].view(b1, 1, -1) - target[None, :, :].view(1, b2, -1), dim=-1, p=2)

        reward, _ = sim_matrix.topk(self.knn_k, dim=1, largest=False, sorted=True)  # (b1, k)

        if not self.knn_avg:  # only keep k-th nearest neighbor
            reward = reward[:, -1]
            reward = reward.reshape(-1, 1)  # (b1, 1)
            if self.rms:
                moving_mean, moving_std = self.disc_state_rms(reward)
                reward = reward / moving_std
            reward = torch.clamp(reward - self.knn_clip, 0)  # (b1, )
        else:  # average over all k nearest neighbors
            reward = reward.reshape(-1, 1)  # (b1 * k, 1)
            if self.rms:
                moving_mean, moving_std = self.disc_state_rms(reward)
                reward = reward / moving_std
            reward = torch.clamp(reward - self.knn_clip, 0)
            reward = reward.reshape((b1, self.knn_k))  # (b1, k)
            reward = reward.mean(dim=1)  # (b1,)
        reward = torch.log(reward + 1.0)
        return reward