from typing import List, Optional

import torch
import torch.nn.functional as F
from torch import nn

# zyq : torch-ngp
import sys
import os

sys.path.append(os.path.abspath(os.path.join(__file__, "..", "..")))



def fc_block(in_f, out_f):
    return torch.nn.Sequential(
        torch.nn.Linear(in_f, out_f),
        torch.nn.ReLU(out_f)
    )

class NeRF(nn.Module):
    def __init__(self, pos_xyz_dim: int,  # 12   positional embedding 阶数
                 pos_dir_dim: int,  # 4 positional embedding 阶数
                 layers: int,  # 8
                 skip_layers: List[int],  # [4]
                 layer_dim: int,  # 256
                 appearance_dim: int,  # 48
                 affine_appearance: bool,  # affine_appearance : False
                 appearance_count: int,  # appearance_count  : number of images (for rubble is 1678)
                 rgb_dim: int,  # rgb_dim : 3
                 xyz_dim: int,  # xyz_dim : fg = 3, bg =4
                 sigma_activation: nn.Module, hparams):
        super(NeRF, self).__init__()


        layer_dim = hparams.layer_dim
        self.xyz_dim = 3
        print(f'pure mlp, layer_dim = {layer_dim}')
        in_channels_xyz = xyz_dim + xyz_dim * pos_xyz_dim * 2
        self.skip_layers = skip_layers
        in_channels_dir = 3 + 3 * pos_dir_dim * 2
        self.embedding_a = nn.Embedding(appearance_count, appearance_dim)
        # output layers
        self.sigma_activation = sigma_activation
        self.rgb_activation = nn.Sigmoid()  # = nn.Sequential(rgb, nn.Sigmoid())

        #semantic
        self.enable_semantic = hparams.enable_semantic
        if self.enable_semantic:
            self.semantic_linear = nn.Sequential(fc_block(layer_dim, layer_dim // 2), nn.Linear(layer_dim // 2, hparams.num_semantic_classes))
            self.semantic_linear_bg = nn.Sequential(fc_block(layer_dim, layer_dim // 2), nn.Linear(layer_dim // 2, hparams.num_semantic_classes))


        #fg
        self.embedding_xyz = Embedding(pos_xyz_dim)
        self.sigma = nn.Linear(layer_dim, 1)
        self.embedding_dir = Embedding(pos_dir_dim)

        xyz_encodings = []
        # xyz encoding layers
        for i in range(layers):
            if i == 0:
                layer = nn.Linear(in_channels_xyz, layer_dim)
            elif i in skip_layers:
                layer = nn.Linear(layer_dim + in_channels_xyz, layer_dim)
            else:
                layer = nn.Linear(layer_dim, layer_dim)
            layer = nn.Sequential(layer, nn.ReLU(True))
            xyz_encodings.append(layer)
        self.xyz_encodings = nn.ModuleList(xyz_encodings)

        self.xyz_encoding_final = nn.Linear(layer_dim, layer_dim)
        # direction and appearance encoding layers
        self.dir_a_encoding = nn.Sequential(
            nn.Linear(layer_dim + in_channels_dir + (appearance_dim if not affine_appearance else 0),
                      layer_dim // 2),
            nn.ReLU(True))

        self.rgb = nn.Linear(
            layer_dim // 2 if (pos_dir_dim > 0 or (appearance_dim > 0 and not affine_appearance)) else layer_dim,
            rgb_dim)

        #bg
        self.embedding_xyz_bg = Embedding(pos_xyz_dim)
        self.sigma_bg = nn.Linear(layer_dim, 1)
        self.embedding_dir_bg = Embedding(pos_dir_dim)

        xyz_encodings_bg = []
        # xyz encoding layers
        for i in range(layers):
            if i == 0:
                layer = nn.Linear(in_channels_xyz, layer_dim)
            elif i in skip_layers:
                layer = nn.Linear(layer_dim + in_channels_xyz, layer_dim)
            else:
                layer = nn.Linear(layer_dim, layer_dim)
            layer = nn.Sequential(layer, nn.ReLU(True))
            xyz_encodings_bg.append(layer)
        self.xyz_encodings_bg = nn.ModuleList(xyz_encodings_bg)
        self.xyz_encoding_final_bg = nn.Linear(layer_dim, layer_dim)

        # direction and appearance encoding layers
        self.dir_a_encoding_bg = nn.Sequential(
            nn.Linear(layer_dim + in_channels_dir + appearance_dim, layer_dim // 2),
            nn.ReLU(True))

        self.rgb_bg = nn.Linear(layer_dim // 2 , rgb_dim)


    def forward(self, point_type, x: torch.Tensor, sigma_only: bool = False,
                    sigma_noise: Optional[torch.Tensor] = None,train_iterations=-1) -> torch.Tensor:

            if point_type == 'fg':
                out = self.forward_fg(point_type, x, sigma_only, sigma_noise,train_iterations=train_iterations)
            elif point_type == 'bg':
                out = self.forward_bg(point_type, x, sigma_only, sigma_noise,train_iterations=train_iterations)
            else:
                NotImplementedError('Unkonwn point type')

            return out

    def forward_fg(self, point_type, x: torch.Tensor, sigma_only: bool = False, sigma_noise: Optional[torch.Tensor] = None,train_iterations=-1) -> torch.Tensor:

        input_xyz = self.embedding_xyz(x[:, :self.xyz_dim])
        xyz_ = input_xyz
        for i, xyz_encoding in enumerate(self.xyz_encodings):
            if i in self.skip_layers:
                xyz_ = torch.cat([input_xyz, xyz_], -1)
            xyz_ = xyz_encoding(xyz_)

        sigma = self.sigma(xyz_)
        if sigma_noise is not None:
            sigma += sigma_noise

        sigma = self.sigma_activation(sigma)

        # semantic 
        if self.enable_semantic:
            sem_logits = self.semantic_linear(xyz_)

        xyz_encoding_final = self.xyz_encoding_final(xyz_)
        dir_a_encoding_input = [xyz_encoding_final]
        dir_a_encoding_input.append(self.embedding_dir(x[:, -4:-1]))
        dir_a_encoding_input.append(self.embedding_a(x[:, -1].long()))
        dir_a_encoding = self.dir_a_encoding(torch.cat(dir_a_encoding_input, -1))
        rgb = self.rgb(dir_a_encoding)

        return torch.cat([self.rgb_activation(rgb), sigma, sem_logits], -1)

    def forward_bg(self, point_type, x: torch.Tensor, sigma_only: bool = False, sigma_noise: Optional[torch.Tensor] = None,train_iterations=-1) -> torch.Tensor:
        input_xyz = self.embedding_xyz_bg(x[:, :self.xyz_dim])
        xyz_ = input_xyz
        for i, xyz_encoding in enumerate(self.xyz_encodings_bg):
            if i in self.skip_layers:
                xyz_ = torch.cat([input_xyz, xyz_], -1)
            xyz_ = xyz_encoding(xyz_)

        sigma = self.sigma_bg(xyz_)

        sigma = self.sigma_activation(sigma)

        # semantic 
        if self.enable_semantic:
            sem_logits = self.semantic_linear_bg(xyz_)

        xyz_encoding_final = self.xyz_encoding_final_bg(xyz_)
        dir_a_encoding_input = [xyz_encoding_final]
        dir_a_encoding_input.append(self.embedding_dir_bg(x[:, -4:-1]))
        dir_a_encoding_input.append(self.embedding_a(x[:, -1].long()))
        dir_a_encoding = self.dir_a_encoding_bg(torch.cat(dir_a_encoding_input, -1))
        rgb = self.rgb_bg(dir_a_encoding)

        return torch.cat([self.rgb_activation(rgb), sigma, sem_logits], -1)


class Embedding(nn.Module):
    def __init__(self, num_freqs: int, logscale=True):
        """
        Defines a function that embeds x to (x, sin(2^k x), cos(2^k x), ...)
        """
        super(Embedding, self).__init__()

        if logscale:
            self.freq_bands = 2 ** torch.linspace(0, num_freqs - 1, num_freqs)
        else:
            self.freq_bands = torch.linspace(1, 2 ** (num_freqs - 1), num_freqs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = [x]
        for freq in self.freq_bands:
            out += [torch.sin(freq * x), torch.cos(freq * x)]

        return torch.cat(out, -1)


class ShiftedSoftplus(nn.Module):
    __constants__ = ['beta', 'threshold']
    beta: int
    threshold: int

    def __init__(self, beta: int = 1, threshold: int = 20) -> None:
        super(ShiftedSoftplus, self).__init__()
        self.beta = beta
        self.threshold = threshold

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.softplus(x - 1, self.beta, self.threshold)

    def extra_repr(self) -> str:
        return 'beta={}, threshold={}'.format(self.beta, self.threshold)