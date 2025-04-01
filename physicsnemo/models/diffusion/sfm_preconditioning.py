# SPDX-FileCopyrightText: Copyright (c) 2023 - 2025 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import importlib
import warnings
from dataclasses import dataclass
from typing import List, Union

import numpy as np
import nvtx
import torch
from torch import nn

from physicsnemo.models.diffusion import DhariwalUNet, SongUNet  # noqa: F401 for globals
from physicsnemo.models.meta import ModelMetaData
from physicsnemo.models.module import Module
from physicsnemo.models.diffusion import Conv2dSerializable

network_module = importlib.import_module("physicsnemo.models.diffusion")

@dataclass
class SFMPrecondSRMetaData(ModelMetaData):
    """EDMPrecondSR meta data"""

    name: str = "SFMPrecondSR"
    # Optimization
    jit: bool = False
    cuda_graphs: bool = False
    amp_cpu: bool = False
    amp_gpu: bool = True
    torch_fx: bool = False
    # Data type
    bf16: bool = False
    # Inference
    onnx: bool = False
    # Physics informed
    func_torch: bool = False
    auto_grad: bool = False


class SFMPrecondSR(Module):
    def __init__(
        self,
        img_resolution: Union[List[int], int],
        img_in_channels: int,
        img_out_channels: int,
        use_fp16: bool = False,
        N_grid_channels: int = 0,
        sigma_max: float = float("inf"),
        sigma_data: float = 0.5,
        model_type: str = "SongUNetPosEmbd",
        use_x_low_conditioning=None,
        **model_kwargs,
    ) -> None:
        """
        preconditioning based on the Stochastic Flow Model approach

        Parameters
        ----------
        img_resolution : Union[List[int], int]
            Image resolution.
        img_in_channels : int
            Number of input color channels.
        img_out_channels : int
            Number of output color channels.
        use_fp16 : bool
            Execute the underlying model at FP16 precision?, by default False.
        sigma_max : float
            Maximum supported noise level, by default inf.
        sigma_data : float
            Expected standard deviation of the training data, by default 0.5.
        model_type :str
            Class name of the underlying model, by default "SongUNetPosEmbd".
        **model_kwargs : dict
            Keyword arguments for the underlying model.
        """
        Module.__init__(self, meta=SFMPrecondSRMetaData)
        model_class = getattr(network_module, model_type)

        self.use_fp16 = use_fp16
        self.sigma_data = sigma_data
        self.img_shape_y = img_resolution[0]
        self.img_shape_x = img_resolution[1]
        self.use_x_low_conditioning = use_x_low_conditioning

        if type(sigma_max) == float:
            self.sigma_max_current = torch.tensor(sigma_max)
        else:    
            sigma_max_current = torch.tensor(sigma_max['initial_values'])
            self.register_buffer('sigma_max_current', sigma_max_current)
            self.ema_weight = torch.tensor(sigma_max['ema_weight'])
            self.min_values = torch.tensor(sigma_max['min_values'])

        # SongUNetPosEmbd
        if 'encoder_type' in model_kwargs:
            del model_kwargs['encoder_type']
        lr_channels = img_in_channels if self.use_x_low_conditioning else 0
        self.denoiser_net = model_class(
            img_resolution=img_resolution,
            in_channels=img_out_channels + lr_channels + N_grid_channels,
            out_channels=img_out_channels,
            **model_kwargs)

    def update_sigma_max(self, sigma_max: float):
        """
        Updates the maximum noise level

        Sigma max is updated externally so any needed accumulation
        and reductions can be handled by the training loop

        Parameters
        ----------
        sigma_max : float
            Maximum noise level
        """
        ema_weight = self.ema_weight.to(self.sigma_max_current.device)
        sigma_max = torch.tensor(sigma_max).to(self.sigma_max_current.device)
        # Update sigma_max_current without gradients
        new_sigma_max_current = ema_weight * self.sigma_max_current + (1 - ema_weight) * sigma_max
        self.sigma_max_current = torch.max(new_sigma_max_current, self.min_values.to(self.sigma_max_current.device))

    def get_sigma_max(self):
        """ returns the current max sigma """
        return self.sigma_max_current    
    
    @nvtx.annotate(message="SFMPrecondSR", color="orange")
    def forward(
        self,
        x: torch.Tensor,
        sigma,
        condition: torch.Tensor,
        force_fp32: bool=False,
        **model_kwargs,
    ):
        """
        Forward pass of the Stochastic Flow Model preconditioner

        Parameters
        ----------
        x : tensor
            The partially noised input image

        sigma : torch.Tensor
            The image containing random noise

        condition : torch.Tensor
            The low resoltuion 

        force_fp32 : bool
            Whether float 32 computations should be forced, default False

        model_kwargs: dict
            Keyword arguments for the underlying model.

        Returns
        -------
        torch.Tensor : the denoised image

        """
        x = x.to(torch.float32)
        dtype = (
            torch.float16
            if (self.use_fp16 and not force_fp32 and x.device.type == "cuda")
            else torch.float32
        )
        c_skip = self.sigma_data**2 / (sigma**2 + self.sigma_data**2)
        c_out = sigma * self.sigma_data / (sigma**2 + self.sigma_data**2).sqrt()
        c_in = 1 / (self.sigma_data**2 + sigma**2).sqrt()
        c_noise = sigma.log() / 4

        if not self.use_x_low_conditioning:
            scaled_x = c_in * x
        else:
            condition = condition.to(torch.float32)
            scaled_x = torch.cat([c_in * x, condition], dim=1)
        
        F_x = self.denoiser_net(
            scaled_x.to(dtype),
            noise_labels = c_noise.flatten(),
            class_labels = None
        )

        D_x = c_skip * x + c_out * F_x.to(torch.float32)
        return D_x


    def round_sigma(self, sigma):
        return torch.as_tensor(sigma)

class SFMPrecondEmpty(Module):
    def __init__(self, **kwargs):
        """
        A preconditioner that does nothing

        Parameters
        ----------
        **model_kwargs : dict
            Keyword arguments for the underlying model.
        """
        super().__init__()
        self.param = torch.nn.Parameter(torch.tensor(0.0))
        self.label_dim = None
