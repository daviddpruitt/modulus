# SPDX-FileCopyrightText: Copyright (c) 2023 - 2024 NVIDIA CORPORATION & AFFILIATES.
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

from collections.abc import Callable
from typing import Dict

import nvtx
import torch
import torch.nn as nn
from omegaconf import DictConfig

from physicsnemo.models.diffusion import SongUNetPosEmbd


def sigma(t):
    return t


def sigma_inv(sigma):
    return sigma


@nvtx.annotate(message="SFM_encoder_sampler", color="red")
def SFM_encoder_sampler(
    networks: Dict[str, torch.nn.Module],
    img_lr: torch.Tensor,
    randn_like: Callable = None,
    cfg: DictConfig = None,
):
    """
    Sampler for the SFM encoder, just runs the encoder

    networks: Dict
        A dictionary containing "encoder_net" and "denoiser_net" entries
        for the denoiser and encoder networks.
        Note: denoiser_net is not used for SFM_encoder_sampler
    img_lr: torch.tensor
        The low resolution image used for denoising
    randn_like: StackedRandomGenerator
        The random noise generator used for denoising.
        Note: not used for SFM_encoder_sampler
    cfg: DictConfig
        The configuration used for sampling
        Note: not used for SFM_encoder_sampler
    """
    encoder_net = networks["encoder_net"]
    x_low = img_lr
    # in V1 the encoder net was inside the denoiser
    if isinstance(encoder_net, SongUNetPosEmbd) or (
        isinstance(encoder_net, nn.parallel.DistributedDataParallel)
        and isinstance(encoder_net.module, SongUNetPosEmbd)
    ):
        x_0 = encoder_net(x_low, noise_labels=torch.tensor([0]), class_labels=None)
    else:
        x_0 = encoder_net(x_low)  # MODULUS

    return x_0


@nvtx.annotate(message="SFM_Euler_sampler", color="red")
def SFM_Euler_sampler(
    networks: Dict[str, torch.nn.Module],
    img_lr: torch.Tensor,
    randn_like: Callable,
    cfg: DictConfig,
):
    """
    Sampler for the SFM encoder, just runs the encoder

    networks: Dict
        A dictionary containing "encoder_net" and "denoiser_net" entries
        for the denoiser and encoder networks.
        Note: denoiser_net is not used for SFM_encoder_sampler
    img_lr: torch.tensor
        The low resolution image used for denoising
    randn_like: StackedRandomGenerator
        The random noise generator used for denoising.
    cfg: DictConfig
        The configuration used for sampling
    """
    denoiser_net = networks["denoiser_net"]
    encoder_net = networks["encoder_net"]

    x_low = img_lr

    # Define time steps in terms of noise level.
    step_indices = torch.arange(cfg.num_steps, device=denoiser_net.device)
    # STATHI TODO: This is a hack, we should treat s_max per channels
    sigma_max = (
        denoiser_net.get_sigma_max()[0]
        if len(denoiser_net.get_sigma_max().shape) > 0
        else denoiser_net.get_sigma_max()
    )
    sigma_steps = (
        sigma_max ** (1 / cfg.rho)
        + step_indices
        / (cfg.num_steps - 1)
        * (cfg.sigma_min ** (1 / cfg.rho) - sigma_max ** (1 / cfg.rho))
    ) ** cfg.rho

    # Define noise level cfg.schedule.
    # if cfg.schedule == "linear":
    #    sigma = lambda t: t
    #    sigma_inv = lambda sigma: sigma

    # Compute final time steps based on the corresponding noise levels.
    t_steps = sigma_inv(denoiser_net.round_sigma(sigma_steps))
    t_steps = torch.cat([t_steps, torch.zeros_like(t_steps[:1])])  # t_N = 0

    # Main sampling loop.
    t_next = t_steps[0]

    if isinstance(encoder_net, SongUNetPosEmbd) or (
        isinstance(encoder_net, nn.parallel.DistributedDataParallel)
        and isinstance(encoder_net.module, SongUNetPosEmbd)
    ):
        x_0 = encoder_net(x_low, noise_labels=torch.tensor([0]), class_labels=None)
    else:
        x_0 = encoder_net(x_low)  # MODULUS

    # x_0 = x_0.to(torch.float64)
    x_t = x_0 + sigma_max * randn_like(x_0)
    for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])):  # 0, ..., N-1
        x_t_cur = x_t
        t_hat = t_cur
        x_t_hat = x_t_cur

        # Euler step.
        x_t_denoised = denoiser_net(x_t_hat, sigma(t_hat), condition=x_low).to(
            torch.float64
        )

        u_t = (x_t_denoised - x_t_hat) / (torch.clamp(t_hat, min=cfg.t_min))

        dt = t_hat - t_next  # needs to be reversed
        x_t = x_t_hat + u_t * dt

    return x_t


@nvtx.annotate(message="SFM_Euler_sampler_Adaptive_Sigma", color="red")
def SFM_Euler_sampler_Adaptive_Sigma(
    networks: Dict[str, torch.nn.Module],
    img_lr: torch.Tensor,
    randn_like: Callable,
    cfg: DictConfig,
):
    """
    Sampler for the SFM encoder, just runs the encoder

    networks: Dict
        A dictionary containing "encoder_net" and "denoiser_net" entries
        for the denoiser and encoder networks.
        Note: denoiser_net is not used for SFM_encoder_sampler
    img_lr: torch.tensor
        The low resolution image used for denoising
    randn_like: StackedRandomGenerator
        The random noise generator used for denoising.
    cfg: DictConfig
        The configuration used for sampling
    """
    denoiser_net = networks["denoiser_net"]
    encoder_net = networks["encoder_net"]

    x_low = img_lr

    # Define time steps in terms of noise level.
    step_indices = torch.arange(cfg.num_steps, device=denoiser_net.device)
    sigma_max_adaptive = denoiser_net.get_sigma_max()

    # set sigma_max for sampling purposes to 1.0, this normalizes time [0-1]
    sigma_max = 1.0
    sigma_steps = (
        sigma_max ** (1 / cfg.rho)
        + step_indices
        / (cfg.num_steps - 1)
        * (cfg.sigma_min ** (1 / cfg.rho) - sigma_max ** (1 / cfg.rho))
    ) ** cfg.rho

    # Define noise level cfg.schedule.
    # if cfg.schedule == "linear":
    #    sigma = lambda t: t
    #    # sigma_deriv = lambda t: 1
    #    sigma_inv = lambda sigma: sigma

    # Compute final time steps based on the corresponding noise levels.
    t_steps = sigma_inv(denoiser_net.round_sigma(sigma_steps))
    t_steps = torch.cat([t_steps, torch.zeros_like(t_steps[:1])])  # t_N = 0

    # Main sampling loop.
    t_next = t_steps[0]

    if isinstance(encoder_net, SongUNetPosEmbd) or (
        isinstance(encoder_net, nn.parallel.DistributedDataParallel)
        and isinstance(encoder_net.module, SongUNetPosEmbd)
    ):
        x_0 = encoder_net(x_low, noise_labels=torch.tensor([0]), class_labels=None)
    else:
        x_0 = encoder_net(x_low)  # MODULUS

    x_t = x_0 + sigma_max_adaptive.view(1, -1, 1, 1) * randn_like(x_0)
    for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])):  # 0, ..., N-1
        x_t_cur = x_t

        t_hat = t_cur
        x_t_hat = x_t_cur

        # Euler step.
        x_t_denoised = denoiser_net(x_t_hat, sigma(t_hat), condition=x_low).to(
            torch.float64
        )

        u_t = (x_t_denoised - x_t_hat) / (torch.clamp(t_hat, min=cfg.t_min))

        dt = t_hat - t_next  # needs to be reversed
        x_t = x_t_hat + u_t * dt

    return x_t
