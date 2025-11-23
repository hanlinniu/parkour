import torch
import torch.nn as nn
import sys
import torchvision

class BEV_RecurrentDepthBackbone(nn.Module):
    def __init__(self) -> None:                
        super().__init__()                                       
        # activation = nn.ELU()
        activation = "elu"
        self.activation = activation
        last_activation = nn.Tanh()
        # self.base_backbone = base_backbone
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.estimator = Estimator(53, 3, hidden_dims=[128, 64], activation="elu").to(self.device)

        print("self.device in bev_depth_backbone.py is ", self.device)

# yaw network######################################################################################################
        # self.cnn_yaw = depth_CNN(output_dim=32, output_activation="tanh").to(self.device)  # num_frames = 2
        self.cnn_yaw = DepthOnlyFCBackbone58x87(53, 32, 512).to(self.device)

        self.combination_mlp_yaw = nn.Sequential(
                                    nn.Linear(32 + 53, 128),
                                    nn.ELU(),
                                    nn.Linear(128, 32)
                                )
        self.rnn_yaw = nn.GRU(input_size=32, hidden_size=512, batch_first=True)
        self.output_mlp_yaw = nn.Sequential(
                                nn.Linear(512, 32+2),
                                nn.Tanh()
                            )
        self.hiddent_states_yaw = None
##################################################################################################################        

        # self.cnn = depth_CNN_GRU(output_dim=32, output_activation="tanh").to(self.device)  # num_frames = 2
        self.history_encoder = StateHistoryEncoder(activation, input_size=53, tsteps=10, output_size=20)

        # self.combination_mlp = nn.Sequential(
        #     nn.Linear(32 + 53, 128),  # 32 (cnn latent) + 53 (proprioception) = 85
        #     nn.ELU(),
        #     nn.Linear(128, 34)
        # )

        # self.rnn = nn.GRU(input_size=34, hidden_size=512, batch_first=True)
        # self.cnn_hidden_states = None
        # self.output_mlp = nn.Sequential(
        #     nn.Linear(512, 32 + 2),
        #     nn.Tanh()
        # )


        # self.head_z_mu = nn.Linear(32 + 3, 32)  # z_mu 32 + 3
        self.hidden_states = None
        self.counter = 0


        print("YESSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSS in depth_backbone.py")      # This is the input of RecurrentDepthBackbone, which is depth_encoder
        # print("base_backbone is : ", base_backbone)

    def forward(self, depth_image, obs):   # this function is used in "depth_latent_and_yaw = depth_encoder(infos["depth"], obs_student)" in play_test.py
        ########################################################################################################
        obs_prop = obs[:, :53]
        obs_history = obs[:, -530:]

        obs_prop_yaw_mask = obs_prop.clone()
        obs_prop_yaw_mask[:, 6:8] = 0.0


        depth_image = depth_image[:, -2, :, :]  
        # abs_vel = self.estimator(obs_prop_yaw_mask)

        ########################################################################################################
        if self.counter % 5 == 0:
            cnn_yaw_latent = self.cnn_yaw(depth_image)
            yaw_latent = self.combination_mlp_yaw(torch.cat((cnn_yaw_latent, obs_prop_yaw_mask), dim=-1))
            yaw_latent, self.hiddent_states_yaw = self.rnn_yaw(yaw_latent[:, None, :], self.hiddent_states_yaw)
            self.hiddent_states_yaw = self.hiddent_states_yaw.detach()
            z_yaw = self.output_mlp_yaw(yaw_latent.squeeze(1))
            z = z_yaw[:, :-2]
            yaw = z_yaw[:, -2:]
            self.yaw = yaw
            self.z = z

        else:
            yaw = self.yaw
            z = self.z


        ########################################################################################################
        obs_prop_yaw = obs_prop.clone()
        obs_prop_yaw[:, 6:8] = yaw*1.5
        abs_vel = self.estimator(obs_prop_yaw)

        depth_latent = z

        priv_latent = self.history_encoder(obs_history.view(-1, 10, 53))

        self.counter += 1


        return depth_latent, abs_vel, priv_latent, yaw

    def detach_hidden_states(self):
        # self.hidden_states = self.hidden_states.detach().clone()
        if self.hidden_states is not None:
            self.hidden_states = self.hidden_states.detach()

    def reset_hidden_state(self, batch_size, device):
        hidden_size = self.rnn.hidden_size
        self.hidden_states = torch.zeros(1, batch_size, hidden_size, device=device)

class depth_CNN(nn.Module):
    """
    Compact depth CNN for inputs shaped [B, 2, 58, 87].
    - Processes each of the 2 frames independently with a small CNN
    - Averages (or concatenates) the frame features (here: average)
    - No GRU, no temporal recurrence
    - Optional micro-batching to reduce peak memory
    """
    def __init__(self, output_dim, output_activation=None, feat_dim=64):
        super().__init__()
        act = nn.ELU()
        self._act_out = nn.Tanh() if output_activation == "tanh" else act

        # Per-frame CNN (same as before)
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

        # Final head (now takes feat_dim instead of GRU hidden state)
        self.head = nn.Linear(feat_dim, output_dim)

    @torch.no_grad()
    def _cnn_chunk(self, frames: torch.Tensor, chunk: int) -> torch.Tensor:
        """Run CNN on frames [N,1,58,87] in smaller micro-batches if needed."""
        if chunk is None or chunk <= 0 or frames.size(0) <= (chunk or 0):
            return self.frame_cnn(frames)

        outs = []
        for i in range(0, frames.size(0), chunk):
            outs.append(self.frame_cnn(frames[i : i + chunk]))
        return torch.cat(outs, dim=0)

    def forward(self, images: torch.Tensor, cnn_microbatch: int = 0):
        """
        images: [B, 2, 58, 87]   (2 stacked frames)
        cnn_microbatch: optional micro-batch size for memory saving

        Returns:
            latent [B, output_dim]
        """
        assert images.dim() == 4 and images.size(1) == 2, \
            f"expected [B,2,58,87], got {tuple(images.shape)}"

        B, T, H, W = images.shape  # T=2
        frames = images.view(B * T, 1, H, W)  # → [B*2, 1, 58, 87]

        # Encode all frames
        f = self._cnn_chunk(frames, cnn_microbatch)       # [B*2, feat_dim]

        # Reshape back to [B,2,feat_dim]
        f = f.view(B, T, -1)

        # Remove temporal structure (simple average)
        fused = f.mean(dim=1)     # [B,feat_dim]

        # Final output
        latent = self.head(fused) # [B,output_dim]
        return self._act_out(latent)

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

class BEV_depth_CNN_GRU(nn.Module):
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
        projection = self.encoder(obs.reshape([nd * T, -1])) # obs size is [36864, 10, 53];    obs.reshape([nd * T, -1]) size is [368640, 53];   projection size is [368640, 30]
        output = self.conv_layers(projection.reshape([nd, T, -1]).permute((0, 2, 1)))  # projection.reshape([nd, T, -1]) size is [3684, 10, 30];  projection.reshape([nd, T, -1]).permute(0,2,1) size is [3684, 30, 10];   output size is [3684, 30]
        output = self.linear_output(output)   
        return output
    


      
class DepthOnlyFCBackbone58x87(nn.Module):
    def __init__(self, prop_dim, scandots_output_dim, hidden_state_dim, output_activation=None, num_frames=1):
        super().__init__()

        self.num_frames = num_frames
        activation = nn.ELU()
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
            nn.Linear(128, scandots_output_dim)
        )

        if output_activation == "tanh":
            self.output_activation = nn.Tanh()
        else:
            self.output_activation = activation

    def forward(self, images: torch.Tensor):
        images_compressed = self.image_compression(images.unsqueeze(1))    # images size is [192, 58, 87].   images.unsqueeze(1) size is [192, 1, 58, 87].  
        latent = self.output_activation(images_compressed)                 # images_compressed size is [192, 32].   latent size is [192, 32]  
        return latent