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

import time
import os
from collections import deque
import statistics

# from torch.utils.tensorboard import SummaryWriter
import torch
import torch.optim as optim
import wandb
# import ml_runlog
import datetime

from rsl_rl.algorithms import PPO
from rsl_rl.modules import *
from rsl_rl.env import VecEnv
import sys
from copy import copy, deepcopy
import warnings

class OnPolicyRunner:

    def __init__(self,
                 env: VecEnv,
                 train_cfg,
                 log_dir=None,
                 init_wandb=True,
                 device='cpu', **kwargs):

        self.cfg=train_cfg["runner"]
        self.alg_cfg = train_cfg["algorithm"]
        self.policy_cfg = train_cfg["policy"]
        self.estimator_cfg = train_cfg["estimator"]
        self.depth_encoder_cfg = train_cfg["depth_encoder"]
        self.device = device
        self.env = env

        print("Using MLP and Priviliged Env encoder ActorCritic structure")
        actor_critic: ActorCritic_DWAQ = ActorCritic_DWAQ(self.env.cfg.env.n_proprio,               # 53
                                                          self.env.num_actions,                     # 12    
                                                          self.env.cfg.env.history_len,             # 10
                                                          **self.policy_cfg).to(self.device)

        # Create algorithm
        alg_class = eval(self.cfg["algorithm_class_name"]) # "algorithm_class_name" is PPO
        self.alg: PPO = alg_class(actor_critic, 
                                #   estimator, self.estimator_cfg, 
                                #   depth_encoder, self.depth_encoder_cfg, depth_actor,
                                  device=self.device, **self.alg_cfg)
        self.num_steps_per_env = self.cfg["num_steps_per_env"]       # 24
        self.save_interval = self.cfg["save_interval"]
        self.dagger_update_freq = self.alg_cfg["dagger_update_freq"]
        # self.dagger_vision_update_freq = self.alg_cfg["dagger_vision_update_freq"]
        # self.update_interval = self.alg_cfg["update_interval"] 

        self.alg.init_storage(
            self.env.num_envs, 
            self.num_steps_per_env,                 # 24
            [self.env.num_obs], 
            [self.env.num_privileged_obs], 
            [self.env.num_actions],
        )

        # self.learn = self.learn_RL if not self.if_depth else self.learn_vision      # if self.if_depth is false, self.learn = self.learn_RL.  otherwise, self.learn = self.learn_vision
        
        self.learn = self.learn_RL


        # Log
        self.log_dir = log_dir
        self.writer = None
        self.tot_timesteps = 0
        self.tot_time = 0
        self.current_learning_iteration = 0

 

    def learn_RL(self, num_learning_iterations, init_at_random_ep_len=False):
        mean_value_loss = 0.
        mean_surrogate_loss = 0.
        mean_estimator_loss = 0.
        mean_disc_loss = 0.
        mean_disc_acc = 0.
        mean_hist_latent_loss = 0.
        mean_priv_reg_loss = 0. 
        priv_reg_coef = 0.
        entropy_coef = 0.
        # initialize writer
        # if self.log_dir is not None and self.writer is None:
        #     self.writer = SummaryWriter(log_dir=self.log_dir, flush_secs=10)
        if init_at_random_ep_len:
            self.env.episode_length_buf = torch.randint_like(self.env.episode_length_buf, high=int(self.env.max_episode_length))
        obs = self.env.get_observations()
        privileged_obs = self.env.get_privileged_observations()              # during training, privileged_obs is None
        critic_obs = privileged_obs if privileged_obs is not None else obs   # critic_obs = obs
        obs, critic_obs = obs.to(self.device), critic_obs.to(self.device)

        infos = {}
        infos["depth"] = self.env.depth_buffer.clone().to(self.device) # if self.if_depth else None
        self.alg.actor_critic.train() # switch to train mode (for dropout for example)


        ep_infos = []
        rewbuffer = deque(maxlen=100)
        rew_explr_buffer = deque(maxlen=100)
        rew_entropy_buffer = deque(maxlen=100)
        lenbuffer = deque(maxlen=100)
        cur_reward_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
        cur_reward_explr_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
        cur_reward_entropy_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
        cur_episode_length = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)

        tot_iter = self.current_learning_iteration + num_learning_iterations
        self.start_learning_iteration = copy(self.current_learning_iteration)

        for it in range(self.current_learning_iteration, tot_iter):                               # self.alg is PPO
            start = time.time()
            hist_encoding = it % self.dagger_update_freq == 0          
           
            # Rollout
            with torch.inference_mode():
                for i in range(self.num_steps_per_env):              # self.num_steps_per_env is 24
                    
                    actions = self.alg.act(obs, critic_obs, infos, hist_encoding=False)    
                    obs, privileged_obs, rewards, dones, infos = self.env.step(actions)  # obs has changed to next_obs !! if done obs has been reset.  privileged_obs is None
                    

                    critic_obs = privileged_obs if privileged_obs is not None else obs
                    obs, critic_obs, rewards, dones = obs.to(self.device), critic_obs.to(self.device), rewards.to(self.device), dones.to(self.device)
                    total_rew = self.alg.process_env_step(rewards, dones, infos)        # self.alg is PPO    # added the transitions in storage, return the reward

                    if self.log_dir is not None:                                                 # True
                        # Book keeping
                        if 'episode' in infos:
                            ep_infos.append(infos['episode'])
                        cur_reward_sum += total_rew
                        cur_reward_explr_sum += 0
                        cur_reward_entropy_sum += 0
                        cur_episode_length += 1
                        
                        new_ids = (dones > 0).nonzero(as_tuple=False)
                        
                        rewbuffer.extend(cur_reward_sum[new_ids][:, 0].cpu().numpy().tolist())
                        rew_explr_buffer.extend(cur_reward_explr_sum[new_ids][:, 0].cpu().numpy().tolist())
                        rew_entropy_buffer.extend(cur_reward_entropy_sum[new_ids][:, 0].cpu().numpy().tolist())
                        lenbuffer.extend(cur_episode_length[new_ids][:, 0].cpu().numpy().tolist())
                        
                        cur_reward_sum[new_ids] = 0
                        cur_reward_explr_sum[new_ids] = 0
                        cur_reward_entropy_sum[new_ids] = 0
                        cur_episode_length[new_ids] = 0
                    

                stop = time.time()
                collection_time = stop - start

                # Learning step
                start = stop
                self.alg.compute_returns(critic_obs)              # using the last critic_obs, and the history batch to recursively calculate the self.returns in the batch
                
            

            # mean_value_loss, mean_surrogate_loss, mean_estimator_loss, mean_discriminator_loss, mean_discriminator_acc, mean_priv_reg_loss = self.alg.update()     # there is actor_critic.act inside
            
            mean_value_loss, mean_surrogate_loss, mean_discriminator_loss, mean_discriminator_acc, mean_autoenc_loss, mean_vel_target_loss, mean_hf_target_loss, mean_obs_prop_target_loss, mean_obs_scan_target_loss, mean_priv_target_loss = self.alg.update()
            # if hist_encoding:
            #     print("Updating dagger...")                           # updating dagger only happens in the Phase 1, update hist_encoder to make it move hist_latent close to priv_latent(constant)  
            #     mean_hist_latent_loss = self.alg.update_dagger()

            stop = time.time()
            learn_time = stop - start
            if self.log_dir is not None:
                self.log(locals())
            if it < 2500:
                if it % self.save_interval == 0:
                    self.save(os.path.join(self.log_dir, 'model_{}.pt'.format(it)))
            elif it < 5000:
                if it % (2*self.save_interval) == 0:
                    self.save(os.path.join(self.log_dir, 'model_{}.pt'.format(it)))
            else:
                if it % (5*self.save_interval) == 0:
                    self.save(os.path.join(self.log_dir, 'model_{}.pt'.format(it)))
            ep_infos.clear()
        
        # self.current_learning_iteration += num_learning_iterations
        self.save(os.path.join(self.log_dir, 'model_{}.pt'.format(self.current_learning_iteration)))


    
    def log_vision(self, locs, width=80, pad=35):
        self.tot_timesteps += self.num_steps_per_env * self.env.num_envs
        self.tot_time += locs['collection_time'] + locs['learn_time']
        iteration_time = locs['collection_time'] + locs['learn_time']

        ep_string = f''
        wandb_dict = {}
        if locs['ep_infos']:
            for key in locs['ep_infos'][0]:
                infotensor = torch.tensor([], device=self.device)
                for ep_info in locs['ep_infos']:
                    # handle scalar and zero dimensional tensor infos
                    if not isinstance(ep_info[key], torch.Tensor):
                        ep_info[key] = torch.Tensor([ep_info[key]])
                    if len(ep_info[key].shape) == 0:
                        ep_info[key] = ep_info[key].unsqueeze(0)
                    infotensor = torch.cat((infotensor, ep_info[key].to(self.device)))
                value = torch.mean(infotensor)
                wandb_dict['Episode_rew/' + key] = value
                ep_string += f"""{f'Mean episode {key}:':>{pad}} {value:.4f}\n"""
        mean_std = self.alg.actor_critic.std.mean()
        fps = int(self.num_steps_per_env * self.env.num_envs / (locs['collection_time'] + locs['learn_time']))

        # wandb_dict['Loss_depth/delta_yaw_ok_percent'] = locs['delta_yaw_ok_percentage']
        # wandb_dict['Loss_depth/depth_encoder'] = locs['depth_encoder_loss']
        # wandb_dict['Loss_depth/depth_actor'] = locs['depth_actor_loss']
        # wandb_dict['Loss_depth/yaw'] = locs['yaw_loss']

        wandb_dict['Policy/mean_noise_std'] = mean_std.item()
        wandb_dict['Perf/total_fps'] = fps
        wandb_dict['Perf/collection time'] = locs['collection_time']
        wandb_dict['Perf/learning_time'] = locs['learn_time']
        if len(locs['rewbuffer']) > 0:
            wandb_dict['Train/mean_reward'] = statistics.mean(locs['rewbuffer'])
            wandb_dict['Train/mean_episode_length'] = statistics.mean(locs['lenbuffer'])
        
        wandb.log(wandb_dict, step=locs['it'])

        str = f" \033[1m Learning iteration {locs['it']}/{self.current_learning_iteration + locs['num_learning_iterations']} \033[0m "

        if len(locs['rewbuffer']) > 0:
            log_string = (f"""{'#' * width}\n"""
                          f"""{str.center(width, ' ')}\n\n"""
                          f"""{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {locs[
                            'collection_time']:.3f}s, learning {locs['learn_time']:.3f}s)\n"""
                          f"""{'Mean action noise std:':>{pad}} {mean_std.item():.2f}\n"""
                          f"""{'Mean reward (total):':>{pad}} {statistics.mean(locs['rewbuffer']):.2f}\n"""
                          f"""{'Mean episode length:':>{pad}} {statistics.mean(locs['lenbuffer']):.2f}\n"""
                          f"""{'Depth encoder loss:':>{pad}} {locs['depth_encoder_loss']:.4f}\n"""
                          f"""{'Depth actor loss:':>{pad}} {locs['depth_actor_loss']:.4f}\n"""
                          f"""{'Yaw loss:':>{pad}} {locs['yaw_loss']:.4f}\n"""
                          f"""{'Delta yaw ok percentage:':>{pad}} {locs['delta_yaw_ok_percentage']:.4f}\n""")
        else:
            log_string = (f"""{'#' * width}\n""")

        log_string += f"""{'-' * width}\n"""
        log_string += ep_string
        curr_it = locs['it'] - self.start_learning_iteration
        eta = self.tot_time / (curr_it + 1) * (locs['num_learning_iterations'] - curr_it)
        mins = eta // 60
        secs = eta % 60
        log_string += (f"""{'-' * width}\n"""
                       f"""{'Total timesteps:':>{pad}} {self.tot_timesteps}\n"""
                       f"""{'Iteration time:':>{pad}} {iteration_time:.2f}s\n"""
                       f"""{'Total time:':>{pad}} {self.tot_time:.2f}s\n"""
                       f"""{'ETA:':>{pad}} {mins:.0f} mins {secs:.1f} s\n""")
        print(log_string)

    def log(self, locs, width=80, pad=35):
        self.tot_timesteps += self.num_steps_per_env * self.env.num_envs
        self.tot_time += locs['collection_time'] + locs['learn_time']
        iteration_time = locs['collection_time'] + locs['learn_time']

        ep_string = f''
        wandb_dict = {}
        if locs['ep_infos']:
            for key in locs['ep_infos'][0]:
                infotensor = torch.tensor([], device=self.device)
                for ep_info in locs['ep_infos']:
                    # handle scalar and zero dimensional tensor infos
                    if not isinstance(ep_info[key], torch.Tensor):
                        ep_info[key] = torch.Tensor([ep_info[key]])
                    if len(ep_info[key].shape) == 0:
                        ep_info[key] = ep_info[key].unsqueeze(0)
                    infotensor = torch.cat((infotensor, ep_info[key].to(self.device)))
                value = torch.mean(infotensor)
                wandb_dict['Episode_rew/' + key] = value
                ep_string += f"""{f'Mean episode {key}:':>{pad}} {value:.4f}\n"""
        mean_std = self.alg.actor_critic.std.mean()
        fps = int(self.num_steps_per_env * self.env.num_envs / (locs['collection_time'] + locs['learn_time']))

        # wandb_dict['Loss/value_function'] = ['mean_value_loss']
        # wandb_dict['Loss/surrogate'] = locs['mean_surrogate_loss']
        # wandb_dict['Loss/estimator'] = locs['mean_estimator_loss']
        # wandb_dict['Loss/hist_latent_loss'] = locs['mean_hist_latent_loss']
        # wandb_dict['Loss/priv_reg_loss'] = locs['mean_priv_reg_loss']
        # wandb_dict['Loss/priv_ref_lambda'] = locs['priv_reg_coef']
        # wandb_dict['Loss/entropy_coef'] = locs['entropy_coef']
        # wandb_dict['Loss/learning_rate'] = self.alg.learning_rate
        # wandb_dict['Loss/discriminator'] = locs['mean_disc_loss']
        # wandb_dict['Loss/discriminator_accuracy'] = locs['mean_disc_acc']


        wandb_dict['Loss/value_function'] = locs['mean_value_loss']
        wandb_dict['Loss/surrogate'] = locs['mean_surrogate_loss']
        # wandb_dict['Loss/estimator'] = locs['mean_estimator_loss']
        # wandb_dict['Loss/hist_latent_loss'] = locs['mean_hist_latent_loss']
        # wandb_dict['Loss/mean_depth_latent_loss'] = locs['mean_depth_latent_loss']  # for depth image
        # wandb_dict['Loss/priv_reg_loss'] = locs['mean_priv_reg_loss']
        # wandb_dict['Loss/priv_ref_lambda'] = locs['priv_reg_coef']

        wandb_dict['Loss/mean_autoenc_loss'] = locs['mean_autoenc_loss']
        wandb_dict['Loss/vel_target'] = locs['mean_vel_target_loss']
        wandb_dict['Loss/hf_target'] = locs['mean_hf_target_loss']
        wandb_dict['Loss/obs_prop_target'] = locs['mean_obs_prop_target_loss']
        wandb_dict['Loss/obs_scan_target'] = locs['mean_obs_scan_target_loss']
        wandb_dict['Loss/priv_target'] = locs['mean_priv_target_loss']
        

        wandb_dict['Loss/entropy_coef'] = locs['entropy_coef']
        wandb_dict['Loss/learning_rate'] = self.alg.learning_rate
        wandb_dict['Loss/discriminator'] = locs['mean_disc_loss']
        wandb_dict['Loss/discriminator_accuracy'] = locs['mean_disc_acc']



        wandb_dict['Policy/mean_noise_std'] = mean_std.item()
        wandb_dict['Perf/total_fps'] = fps
        wandb_dict['Perf/collection time'] = locs['collection_time']
        wandb_dict['Perf/learning_time'] = locs['learn_time']
        if len(locs['rewbuffer']) > 0:
            wandb_dict['Train/mean_reward'] = statistics.mean(locs['rewbuffer'])
            wandb_dict['Train/mean_reward_explr'] = statistics.mean(locs['rew_explr_buffer'])
            wandb_dict['Train/mean_reward_task'] = wandb_dict['Train/mean_reward'] - wandb_dict['Train/mean_reward_explr']
            wandb_dict['Train/mean_reward_entropy'] = statistics.mean(locs['rew_entropy_buffer'])
            wandb_dict['Train/mean_episode_length'] = statistics.mean(locs['lenbuffer'])
            # wandb_dict['Train/mean_reward/time', statistics.mean(locs['rewbuffer']), self.tot_time)
            # wandb_dict['Train/mean_episode_length/time', statistics.mean(locs['lenbuffer']), self.tot_time)

        wandb.log(wandb_dict, step=locs['it'])

        str = f" \033[1m Learning iteration {locs['it']}/{self.current_learning_iteration + locs['num_learning_iterations']} \033[0m "

        if len(locs['rewbuffer']) > 0:
            log_string = (f"""{'#' * width}\n"""
                          f"""{str.center(width, ' ')}\n\n"""
                          f"""{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {locs[
                            'collection_time']:.3f}s, learning {locs['learn_time']:.3f}s)\n"""
                          f"""{'Value function loss:':>{pad}} {locs['mean_value_loss']:.4f}\n"""
                          f"""{'Surrogate loss:':>{pad}} {locs['mean_surrogate_loss']:.4f}\n"""
                        #   f"""{'Discriminator loss:':>{pad}} {locs['mean_disc_loss']:.4f}\n"""
                        #   f"""{'Discriminator accuracy:':>{pad}} {locs['mean_disc_acc']:.4f}\n"""
                          f"""{'Mean action noise std:':>{pad}} {mean_std.item():.2f}\n"""
                          f"""{'Mean reward (total):':>{pad}} {statistics.mean(locs['rewbuffer']):.2f}\n"""
                          f"""{'Mean reward (task):':>{pad}} {statistics.mean(locs['rewbuffer']) - statistics.mean(locs['rew_explr_buffer']):.2f}\n"""
                          f"""{'Mean reward (exploration):':>{pad}} {statistics.mean(locs['rew_explr_buffer']):.2f}\n"""
                          f"""{'Mean reward (entropy):':>{pad}} {statistics.mean(locs['rew_entropy_buffer']):.2f}\n"""
                          f"""{'Mean episode length:':>{pad}} {statistics.mean(locs['lenbuffer']):.2f}\n""")
                        #   f"""{'Mean reward/step:':>{pad}} {locs['mean_reward']:.2f}\n"""
                        #   f"""{'Mean episode length/episode:':>{pad}} {locs['mean_trajectory_length']:.2f}\n""")
        else:
            log_string = (f"""{'#' * width}\n"""
                          f"""{str.center(width, ' ')}\n\n"""
                          f"""{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {locs[
                            'collection_time']:.3f}s, learning {locs['learn_time']:.3f}s)\n"""
                          f"""{'Value function loss:':>{pad}} {locs['mean_value_loss']:.4f}\n"""
                          f"""{'Surrogate loss:':>{pad}} {locs['mean_surrogate_loss']:.4f}\n"""
                        #   f"""{'Estimator loss:':>{pad}} {locs['mean_estimator_loss']:.4f}\n"""
                          f"""{'Estimator loss:':>{pad}} {locs['mean_autoenc_loss']:.4f}\n"""
                          f"""{'Mean action noise std:':>{pad}} {mean_std.item():.2f}\n""")
                        #   f"""{'Mean reward/step:':>{pad}} {locs['mean_reward']:.2f}\n"""
                        #   f"""{'Mean episode length/episode:':>{pad}} {locs['mean_trajectory_length']:.2f}\n""")

        log_string += f"""{'-' * width}\n"""
        log_string += ep_string
        curr_it = locs['it'] - self.start_learning_iteration
        eta = self.tot_time / (curr_it + 1) * (locs['num_learning_iterations'] - curr_it)
        mins = eta // 60
        secs = eta % 60
        log_string += (f"""{'-' * width}\n"""
                       f"""{'Total timesteps:':>{pad}} {self.tot_timesteps}\n"""
                       f"""{'Iteration time:':>{pad}} {iteration_time:.2f}s\n"""
                       f"""{'Total time:':>{pad}} {self.tot_time:.2f}s\n"""
                       f"""{'ETA:':>{pad}} {mins:.0f} mins {secs:.1f} s\n""")
        print(log_string)

    def save(self, path, infos=None):
        state_dict = {
            'model_state_dict': self.alg.actor_critic.state_dict(),
            # 'estimator_state_dict': self.alg.estimator.state_dict(),
            'optimizer_state_dict': self.alg.optimizer.state_dict(),
            'iter': self.current_learning_iteration,
            'infos': infos,
            }
        # if self.if_depth:
        #     state_dict['depth_encoder_state_dict'] = self.alg.depth_encoder.state_dict()
        #     state_dict['depth_actor_state_dict'] = self.alg.depth_actor.state_dict()
        torch.save(state_dict, path)

    def load(self, path, load_optimizer=True):
        print("*" * 80)
        print("Loading model from {}...".format(path))
        loaded_dict = torch.load(path, map_location=self.device)
        self.alg.actor_critic.load_state_dict(loaded_dict['model_state_dict'])     # actor_critic model includes both actor and critic models
        # self.alg.estimator.load_state_dict(loaded_dict['estimator_state_dict'])
        # if self.if_depth:
        #     if 'depth_encoder_state_dict' not in loaded_dict:
        #         warnings.warn("'depth_encoder_state_dict' key does not exist, not loading depth encoder...")
        #     else:
        #         print("Saved depth encoder detected, loading...")
        #         self.alg.depth_encoder.load_state_dict(loaded_dict['depth_encoder_state_dict'])
        #     if 'depth_actor_state_dict' in loaded_dict:
        #         print("Saved depth actor detected, loading...")
        #         self.alg.depth_actor.load_state_dict(loaded_dict['depth_actor_state_dict'])
        #     else:
        #         print("No saved depth actor, Copying actor critic actor to depth actor...")
        #         self.alg.depth_actor.load_state_dict(self.alg.actor_critic.actor.state_dict())
        if load_optimizer:
            self.alg.optimizer.load_state_dict(loaded_dict['optimizer_state_dict'])
        # self.current_learning_iteration = loaded_dict['iter']
        print("*" * 80)
        return loaded_dict['infos']

    def get_inference_policy(self, device=None):
        self.alg.actor_critic.eval() # switch to evaluation mode (dropout for example)
        if device is not None:
            self.alg.actor_critic.to(device)
        return self.alg.actor_critic.act_inference
    
    # def get_depth_actor_inference_policy(self, device=None):
    #     self.alg.depth_actor.eval() # switch to evaluation mode (dropout for example)
    #     if device is not None:
    #         self.alg.depth_actor.to(device)
    #     return self.alg.depth_actor
    
    def get_actor_critic(self, device=None):
        self.alg.actor_critic.eval() # switch to evaluation mode (dropout for example)
        if device is not None:
            self.alg.actor_critic.to(device)
        return self.alg.actor_critic
    
    # def get_estimator_inference_policy(self, device=None):
    #     self.alg.estimator.eval() # switch to evaluation mode (dropout for example)
    #     if device is not None:
    #         self.alg.estimator.to(device)
    #     return self.alg.estimator.inference

    # def get_depth_encoder_inference_policy(self, device=None):
    #     self.alg.depth_encoder.eval()
    #     if device is not None:
    #         self.alg.depth_encoder.to(device)
    #     return self.alg.depth_encoder
    
    # def get_disc_inference_policy(self, device=None):
    #     self.alg.discriminator.eval() # switch to evaluation mode (dropout for example)
    #     if device is not None:
    #         self.alg.discriminator.to(device)
    #     return self.alg.discriminator.inference
