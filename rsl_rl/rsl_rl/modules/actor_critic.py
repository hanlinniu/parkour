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

import numpy as np

import code
import torch
import torch.nn as nn
from torch.distributions import Normal
from torch.nn.modules import rnn
from torch.nn.modules.activation import ReLU

# import depth_backbone
# from .depth_backbone import DepthOnlyFCBackbone58x87_single_depth_encoder


class StateHistoryEncoder(nn.Module):
    def __init__(self, activation_fn, input_size, tsteps, output_size, tanh_encoder_output=False):
        # self.device = device
        super(StateHistoryEncoder, self).__init__()
        self.activation_fn = activation_fn
        self.tsteps = tsteps

        channel_size = 10
        # last_activation = nn.ELU()


        if activation_fn == "elu":
            self.activation_fn = nn.ELU()
        elif activation_fn == "relu":
            self.activation_fn = nn.ReLU()
        elif activation_fn == "tanh":
            self.activation_fn = nn.Tanh()
        else:
            raise ValueError(f"Unsupported activation: {activation_fn}")
        

        self.encoder = nn.Sequential(
                nn.Linear(input_size, 3 * channel_size), self.activation_fn,
                )

        if tsteps == 50:
            self.conv_layers = nn.Sequential(
                    nn.Conv1d(in_channels = 3 * channel_size, out_channels = 2 * channel_size, kernel_size = 8, stride = 4), self.activation_fn,
                    nn.Conv1d(in_channels = 2 * channel_size, out_channels = channel_size, kernel_size = 5, stride = 1), self.activation_fn,
                    nn.Conv1d(in_channels = channel_size, out_channels = channel_size, kernel_size = 5, stride = 1), self.activation_fn, nn.Flatten())
        elif tsteps == 10:
            self.conv_layers = nn.Sequential(
                nn.Conv1d(in_channels = 3 * channel_size, out_channels = 2 * channel_size, kernel_size = 4, stride = 2), self.activation_fn,     # [3684, 20, 4]
                nn.Conv1d(in_channels = 2 * channel_size, out_channels = channel_size, kernel_size = 2, stride = 1), self.activation_fn,         # [3684, 10, 3]
                nn.Flatten())                                                                                                                    # [3684, 30]
        elif tsteps == 20:
            self.conv_layers = nn.Sequential(
                nn.Conv1d(in_channels = 3 * channel_size, out_channels = 2 * channel_size, kernel_size = 6, stride = 2), self.activation_fn,
                nn.Conv1d(in_channels = 2 * channel_size, out_channels = channel_size, kernel_size = 4, stride = 2), self.activation_fn,
                nn.Flatten())
        else:
            raise(ValueError("tsteps must be 10, 20 or 50"))

        self.linear_output = nn.Sequential(
                nn.Linear(channel_size * 3, output_size), self.activation_fn
                )

    def forward(self, obs):
        # nd * T * n_proprio
        nd = obs.shape[0]             # nd=36864
        T = self.tsteps               # 10
        # print("obs device", obs.device)
        # print("encoder device", next(self.encoder.parameters()).device)
        projection = self.encoder(obs.reshape([nd * T, -1])) # obs size is [36864, 10, 53];    obs.reshape([nd * T, -1]) size is [368640, 53];   projection size is [368640, 30]
        output = self.conv_layers(projection.reshape([nd, T, -1]).permute((0, 2, 1)))  # projection.reshape([nd, T, -1]) size is [3684, 10, 30];  projection.reshape([nd, T, -1]).permute(0,2,1) size is [3684, 30, 10];   output size is [3684, 30]
        output = self.linear_output(output)   
        return output


class depth_CNN(nn.Module):
    def __init__(self, output_dim, output_activation=None, num_frames=2):
        super().__init__()

        self.num_frames = num_frames
        self.num_frames = 2
        activation = nn.ELU()

        # self.image_compression = nn.Sequential(
        #     nn.Conv2d(in_channels=self.num_frames, out_channels=16, kernel_size=3, padding=1),  # [16, 58, 87]
        #     nn.MaxPool2d(kernel_size=2, stride=2),                                              # [16, 29, 43]
        #     activation,
        #     nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1),              # [32, 29, 43]
        #     nn.MaxPool2d(kernel_size=2, stride=2),                                              # [32, 14, 21]
        #     activation,
        #     nn.AdaptiveAvgPool2d((1, 1)),                                                       # [32, 1, 1]
        #     nn.Flatten(),                                                                       # [32]
        #     nn.Linear(32, 64),
        #     activation,
        #     nn.Linear(64, output_dim)
        # )

        self.image_compression = nn.Sequential(
            # [1, 58, 87]
            nn.Conv2d(in_channels=self.num_frames, out_channels=32, kernel_size=5),
            # [32, 54, 83]
            nn.MaxPool2d(kernel_size=2, stride=2),
            # [32, 27, 41]
            activation,
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3),
            activation,
            nn.Flatten(),
            # [32, 25, 39]
            nn.Linear(64 * 25 * 39, 128),
            activation,
            nn.Linear(128, output_dim)
        )

        if output_activation == "tanh":
            self.output_activation = nn.Tanh()
        else:
            self.output_activation = activation

    def forward(self, images: torch.Tensor):
        # print("images shape is :", images.shape)  # images shape is : torch.Size([1000, 2, 58, 87])
        images_compressed = self.image_compression(images)
        # print("after image_compression:", images_compressed.shape)
        latent = self.output_activation(images_compressed)               
        return latent
    



class depth_CNN_GRU(nn.Module):
    """
    Memory-lean depth CNN + GRU for images shaped [B, 2, 58, 87].
    Key changes to cut memory:
      - Smaller channel sizes
      - Strided convs + AdaptiveAvgPool2d(1) (no huge Flatten->Linear)
      - Tiny GRU (64 hidden)
      - Optional micro-batching for the CNN stage to cap peak memory
    """
    def __init__(self, output_dim, output_activation=None, feat_dim=64, gru_hidden=64):
        super().__init__()
        act = nn.ELU()
        self._act_out = nn.Tanh() if output_activation == "tanh" else act

        # Per-frame CNN (very small):
        # 1x58x87 -> 16x29x44 -> 32x15x22 -> 48x8x11 -> GAP -> 48 -> FC 64
        self.frame_cnn = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1),  # [B,16,29,44]
            act,
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1), # [B,32,15,22]
            act,
            nn.Conv2d(32, 48, kernel_size=3, stride=2, padding=1), # [B,48,8,11]
            act,
            nn.AdaptiveAvgPool2d(1),                               # [B,48,1,1]
            nn.Flatten(),                                          # [B,48]
            nn.Linear(48, feat_dim),                               # [B,feat_dim]
            act,
        )

        # Tiny GRU across the 2 time steps
        self.gru = nn.GRU(input_size=feat_dim, hidden_size=gru_hidden, num_layers=1, batch_first=True)
        self.head = nn.Linear(gru_hidden, output_dim)

    @torch.no_grad()
    def _cnn_chunk(self, frames: torch.Tensor, chunk: int) -> torch.Tensor:
        """Run frame_cnn on [N,1,58,87] in chunks to reduce peak memory."""
        if chunk is None or chunk <= 0 or frames.size(0) <= (chunk or 0):
            return self.frame_cnn(frames)
        outs = []
        for i in range(0, frames.size(0), chunk):
            outs.append(self.frame_cnn(frames[i:i+chunk]))
        return torch.cat(outs, dim=0)

    def forward(self, images: torch.Tensor, cnn_microbatch: int = 0):
        """
        images: [B, 2, 58, 87] (T=2 stacked as channels)
        cnn_microbatch: if >0, splits the per-frame CNN over micro-batches to save memory
        Returns: latent [B, output_dim]
        """
        assert images.dim() == 4 and images.size(1) == 2, f"expected [B,2,58,87], got {tuple(images.shape)}"
        B, T, H, W = images.shape
        frames = images.view(B * T, 1, H, W)  # [B*T,1,58,87]

        # Encode frames with optional micro-batching
        f = self._cnn_chunk(frames, cnn_microbatch)    # [B*T, feat_dim]
        f = f.view(B, T, -1)                           # [B,2,feat_dim]

        out, _ = self.gru(f)                           # [B,2,gru_hidden]
        last = out[:, -1, :]                           # [B,gru_hidden]
        latent = self.head(last)                       # [B,output_dim]
        return self._act_out(latent)
    

    # def detach_hidden_states(self):
    #     # self.hidden_states = self.hidden_states.detach().clone()
    #     if self.hidden_states is not None:
    #         self.hidden_states = self.hidden_states.detach()



class ActorCritic_DWAQ(nn.Module):
    def __init__(self, num_prop, 
                 num_actions,
                 num_hist,                      # 10 ELU(alpha=1.0)
                 activation='elu',
                 init_noise_std=1.0,
                 **kwargs):
        
        super().__init__()
        hist_prop_num = 530

        critic_input_dim = 753                             # including obs, height (scan_dot), priv_explicit, priv_latent, and obs_history
        num_scan = 132
        self.activation = activation


        actor_input_dim = num_prop  + 32 + 3 + 20


        if activation == "elu":
            self.activation = nn.ELU()
        elif activation == "relu":
            self.activation = nn.ReLU()
        elif activation == "tanh":
            self.activation = nn.Tanh()
        else:
            raise ValueError(f"Unsupported activation: {activation}")



        # self.proprio_mlp = StateHistoryEncoder(activation, input_size=53, tsteps=10, output_size=32)
        self.history_encoder = StateHistoryEncoder(activation, input_size=53, tsteps=10, output_size=20)

        self.priv_encoder =  nn.Sequential(
            nn.Linear(29, 64),
            self.activation,
            nn.Linear(64, 20),
            nn.ELU()
        )   # for processing the mass, friction, motor strength


        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.cnn = depth_CNN_GRU(output_dim=32, output_activation="tanh").to(self.device)  # num_frames = 2

        self.scan_decoder = nn.Sequential(
            nn.Linear(32, 64),
            self.activation,
            nn.Linear(64, 256),
            self.activation,
            nn.Linear(256, 132)  # num_scan = 132
        )

        self.head_z_mu = nn.Linear(32 + 3, 32)  # z_mu 32 + 3
        self.head_z_logvar = nn.Linear(32 + 3, 32)  # z_logvar 32 + 3


        self.combination_mlp = nn.Sequential(
            nn.Linear(32 + 53, 128),  # 32 (cnn latent) + 53 (proprioception) = 85
            nn.ELU(),
            nn.Linear(128, 34)
        )

        self.yaw_decoder = nn.Sequential(
            nn.Linear(34, 16),
            nn.ELU(),
            nn.Linear(16, 2),
            nn.Tanh()
        )

        self.rnn = nn.GRU(
            input_size=34,
            hidden_size=512,
            batch_first=True
        )

        self.output_mlp = nn.Sequential(
            nn.Linear(512, 32 + 2),
            nn.Tanh()
        )


        self.actor = nn.Sequential(
            nn.Linear(actor_input_dim,512),
            self.activation,
            nn.Linear(512,256),
            self.activation,
            nn.Linear(256,128),
            self.activation,
            nn.Linear(128,num_actions)
        )

        self.critic = nn.Sequential(
            nn.Linear(critic_input_dim,512),
            self.activation,
            nn.Linear(512,256),
            self.activation,
            nn.Linear(256,128),
            self.activation,
            nn.Linear(128,1)
        )


       ################################################################################################################
        self.hidden_states = None

        self.cnn_hidden_states = None

        # Action noise
        self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        self.distribution = None
        # disable args validation for speedup
        Normal.set_default_validate_args = False

        self.counter = 0
        self.counter_inference = 0
        

    def forward(self):
        raise NotImplementedError

    def reparameterise(self,mean,logvar):
        var = torch.exp(logvar*0.5)
        code_temp = torch.randn_like(var)
        code = mean + var*code_temp
        return code
    

    def infer_priv_latent(self, priv):
        return self.priv_encoder(priv)
    
    def infer_hist_latent(self, obs_history):
        hist = obs_history
        return self.history_encoder(hist.view(-1, 10, 53))    #  hist.size size is [3684, 530];   hist.view(-1, self.num_hist, self.num_prop) size is [3684, 10, 53]
    
    def reset_cnn_latent(self):
        self.cnn_latent = None

    def cenet_forward(self, abs_vel, priv, image_obs, obs_history, hist_encoding: bool, gen_data):

        if hist_encoding:
            priv_latent = self.infer_hist_latent(obs_history)
        else:
            priv_latent = self.infer_priv_latent(priv)  # priv is the mass, friction, motor strength, output size is [B, 20]

        # print("*" * 50)
        if self.counter % 5 == 0 and gen_data == True:
            # print(f"[step {self.counter}] image_obs sum: {image_obs.sum().item()}")
            self.cnn_latent = self.cnn(image_obs)
            cnn_latent = self.cnn_latent
        elif self.counter % 5 != 0 and gen_data == True:
            # print(f"[step {self.counter}] image_obs sum: {image_obs.sum().item()}")
            cnn_latent = self.cnn_latent
        else:
            cnn_latent = self.cnn(image_obs)

        if gen_data == True:
            self.counter += 1  

        return priv_latent, cnn_latent


    def cenet_forward_inference(self, abs_vel, priv, image_obs, obs_history, hist_encoding: bool, gen_data):

        if hist_encoding:
            priv_latent = self.infer_hist_latent(obs_history)
        else:
            priv_latent = self.infer_priv_latent(priv)  # priv is the mass, friction, motor strength, output size is [B, 20]

        # print("*" * 50)
        if self.counter_inference % 5 == 0 and gen_data == True:
            self.cnn_latent = self.cnn(image_obs)
            cnn_latent = self.cnn_latent
        elif self.counter_inference % 5 != 0 and gen_data == True:
            cnn_latent = self.cnn_latent
        else:
            cnn_latent = self.cnn(image_obs)

        # if gen_data == True:
        self.counter_inference += 1  

        return priv_latent, cnn_latent

    def detach_hidden_states(self):
        if self.hidden_states is not None:
            self.hidden_states = self.hidden_states.detach()

    def reset(self, dones=None):
        pass

    @property
    def action_mean(self):
        return self.distribution.mean

    @property
    def action_std(self):
        return self.distribution.stddev

    @property
    def entropy(self):
        return self.distribution.entropy().sum(dim=-1)
    

    def update_distribution(self, observations):
        mean = self.actor(observations)
        self.distribution = Normal(mean, mean * 0.0 + self.std)



    def act(self, obs, image_obs, hist_encoding, gen_data, **kwargs):
        obs_prop = obs[:, :53] 
        obs_history = obs[:, -530:]
        abs_vel = obs[:, 185:188]  # abs_vel is the 186th to 188th elements of obs_batch
        priv = obs[:, 194:223] # priv is the 195th to 224th elements of obs_batch, which is the mass, friction, motor strength
        obs_scan = obs[:, 53:185]  # obs_scan is the 54th to 185th elements of obs_batch, which is the scan dot

        obs_prop_yaw_mask = obs_prop.clone()
        obs_prop_yaw_mask[:, 6:8] = 0.0  # mask the yaw in obs_prop


        if self.counter % 5 == 0 and gen_data == True:
            self.cnn_latent = self.cnn(image_obs)
            self.obs_prop_yaw_mask = obs_prop_yaw_mask
            cnn_latent = self.cnn_latent
            depth_latent = self.combination_mlp(torch.cat((cnn_latent, self.obs_prop_yaw_mask), dim=-1))
            if self.cnn_hidden_states is None or self.cnn_hidden_states.size(1) != depth_latent.size(0):
                    self.cnn_hidden_states = torch.zeros(
                        1, depth_latent.size(0), self.rnn.hidden_size, device=depth_latent.device
                    )
            depth_latent, self.cnn_hidden_states = self.rnn(depth_latent[:, None, :], self.cnn_hidden_states)
            self.cnn_hidden_states = self.cnn_hidden_states.detach()
            depth_latent = self.output_mlp(depth_latent.squeeze(1))
            self.depth_latent = depth_latent

            combination_cnn_vel = torch.cat((depth_latent[:, :-2], abs_vel), dim=-1)
            z_mu, z_logvar = self.head_z_mu(combination_cnn_vel), self.head_z_logvar(combination_cnn_vel)
            z = self.reparameterise(z_mu, z_logvar)
            decoded_scan = self.scan_decoder(z)
            self.z = z
            self.decoded_scan = decoded_scan

        elif self.counter % 5 != 0 and gen_data == True:
            depth_latent = self.depth_latent

            combination_cnn_vel = torch.cat((depth_latent[:, :-2], abs_vel), dim=-1)
            z_mu, z_logvar = self.head_z_mu(combination_cnn_vel), self.head_z_logvar(combination_cnn_vel)
            z = self.reparameterise(z_mu, z_logvar)
            decoded_scan = self.scan_decoder(z)
            self.z = z
            self.decoded_scan = decoded_scan

        else:
            cnn_latent = self.cnn(image_obs)
            depth_latent = self.combination_mlp(torch.cat((cnn_latent, obs_prop_yaw_mask), dim=-1))
            if self.cnn_hidden_states is None or self.cnn_hidden_states.size(1) != depth_latent.size(0):
                    self.cnn_hidden_states = torch.zeros(
                        1, depth_latent.size(0), self.rnn.hidden_size, device=depth_latent.device
                    )
            depth_latent, self.cnn_hidden_states = self.rnn(depth_latent[:, None, :], self.cnn_hidden_states)
            self.cnn_hidden_states = self.cnn_hidden_states.detach()
            depth_latent = self.output_mlp(depth_latent.squeeze(1))

            combination_cnn_vel = torch.cat((depth_latent[:, :-2], abs_vel), dim=-1)
            z_mu, z_logvar = self.head_z_mu(combination_cnn_vel), self.head_z_logvar(combination_cnn_vel)
            z = self.reparameterise(z_mu, z_logvar)
            decoded_scan = self.scan_decoder(z)

        if gen_data == True:
            self.counter += 1  


        if hist_encoding:
            priv_latent = self.infer_hist_latent(obs_history)
        else:
            priv_latent = self.infer_priv_latent(priv)  # priv is the mass, friction, motor strength, output size is [B, 20]

        observations = torch.cat((obs_prop, depth_latent[:, :-2], abs_vel, priv_latent), dim=-1)   # dims is 53 + 32 + 20 = 105


        self.update_distribution(observations)
        return self.distribution.sample(), decoded_scan


    def get_actions_log_prob(self, actions):
        return self.distribution.log_prob(actions).sum(dim=-1)
    
    
    @torch.inference_mode()
    def act_inference(self, obs, image_obs):
        obs_prop = obs[:, :53] 
        obs_history = obs[:, -530:]
        abs_vel = obs[:, 185:188]  # abs_vel is the 186th to 188th elements of obs_batch
        priv = obs[:, 194:223] # priv is the 195th to 224th elements of obs_batch, which is the mass, friction, motor strength
        obs_scan = obs[:, 53:185]  # obs_scan is the 54th to 185th elements of obs_batch, which is the scan dot

        obs_prop_yaw_mask = obs_prop.clone()
        obs_prop_yaw_mask[:, 6:8] = 0.0  # mask the yaw in obs_prop


        if self.counter % 5 == 0 :
            self.cnn_latent = self.cnn(image_obs)
            self.obs_prop_yaw_mask = obs_prop_yaw_mask
            cnn_latent = self.cnn_latent
            depth_latent = self.combination_mlp(torch.cat((cnn_latent, self.obs_prop_yaw_mask), dim=-1))
            if self.cnn_hidden_states is None or self.cnn_hidden_states.size(1) != depth_latent.size(0):
                    self.cnn_hidden_states = torch.zeros(
                        1, depth_latent.size(0), self.rnn.hidden_size, device=depth_latent.device
                    )
            depth_latent, self.cnn_hidden_states = self.rnn(depth_latent[:, None, :], self.cnn_hidden_states)
            self.cnn_hidden_states = self.cnn_hidden_states.detach()
            depth_latent = self.output_mlp(depth_latent.squeeze(1))
            self.depth_latent = depth_latent

            combination_cnn_vel = torch.cat((depth_latent[:, :-2], abs_vel), dim=-1)
            z = self.head_z_mu(combination_cnn_vel)


        elif self.counter % 5 != 0:

            depth_latent = self.depth_latent

            combination_cnn_vel = torch.cat((depth_latent[:, :-2], abs_vel), dim=-1)
            z = self.head_z_mu(combination_cnn_vel)
            

        self.counter += 1  

        priv_latent = self.infer_hist_latent(obs_history)

        observations = torch.cat((obs_prop, depth_latent[:, :-2], abs_vel, priv_latent), dim=-1)  # dims is 53 + 32 + 20 = 105

        actions_mean = self.actor(observations)
        return actions_mean
    
    
    def evaluate(self, critic_observations, **kwargs):
        value = self.critic(critic_observations)
        return value

    def reset_std(self, std, num_actions, device):
        new_std = std * torch.ones(num_actions, device=device)
        self.std.data = new_std.data