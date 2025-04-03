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

import torch
from omegaconf import DictConfig

from physicsnemo.models.diffusion import SFMPrecondSR, SongUNetPosEmbd
from physicsnemo.utils.generative import (
    SFM_encoder_sampler,
    SFM_Euler_sampler,
    SFM_Euler_sampler_Adaptive_Sigma,
    StackedRandomGenerator,
)


def get_songunet():
    """helper that creates a songunet for testing"""
    songunet_kwargs = {
        "img_resolution": 32,
        "in_channels": 6,
        "out_channels": 6,
        "embedding_type": "zero",
        "label_dim": 0,
        "encoder_type": "standard",
        "decoder_type": "standard",
        "channel_mult_noise": 1,
        "resample_filter": [1, 1],
        "channel_mult": [1, 2, 2],
        "attn_resolutions": [28],
        "N_grid_channels": 0,
        "model_channels": 4,
    }
    return SongUNetPosEmbd(**songunet_kwargs)


class fake_net(torch.nn.Module):
    """dummy class to test sfm encoder"""

    def __init__(self, sigma_max=[0.5]):
        self.sigma_max = torch.tensor(sigma_max)

    def get_sigma_max(self):
        return self.sigma_max

    def round_sigma(self, x):
        return torch.tensor(x)

    def forward(self, x, *args, **kwargs):
        return x


def test_sfm_encoder_sampler():
    """SFM_encoder_sampler"""
    dummy_net = torch.nn.Identity()
    encoder = get_songunet()

    image_rnd = torch.rand((2, 6, 8, 8))

    networks = {
        "encoder_net": dummy_net,
        "denoiser_net": None,
    }

    out_val = SFM_encoder_sampler(networks, image_rnd)
    assert torch.equal(out_val, image_rnd)

    networks = {
        "encoder_net": encoder,
        "denoiser_net": None,
    }
    out_val = SFM_encoder_sampler(networks, image_rnd)
    assert isinstance(out_val, torch.Tensor)


def test_sfm_euler_sampler():
    """SFM_Euler_sampler"""
    encoder = get_songunet()
    denoiser = SFMPrecondSR(
        img_resolution=[32, 32],
        img_in_channels=4,
        img_out_channels=6,
        N_grid_channels=4,
    ).to("cpu")
    dummy_net = torch.nn.Identity()

    batch_seeds = torch.as_tensor([1, 2]).to("cpu")
    image_rnd = torch.rand((2, 6, 32, 32)).to("cpu")
    rnd = StackedRandomGenerator(image_rnd.device, batch_seeds)

    cfg = {"rho": 7, "num_steps": 5, "sigma_min": 0.01, "t_min": 0.002}
    cfg = DictConfig(cfg)

    networks = {
        "encoder_net": dummy_net,
        "denoiser_net": denoiser,
    }

    out_val = SFM_Euler_sampler(
        networks=networks, img_lr=image_rnd, randn_like=rnd.randn_like, cfg=cfg
    )
    assert isinstance(out_val, torch.Tensor)

    networks = {
        "encoder_net": encoder,
        "denoiser_net": denoiser,
    }

    out_val = SFM_Euler_sampler(
        networks=networks, img_lr=image_rnd, randn_like=rnd.randn_like, cfg=cfg
    )
    assert isinstance(out_val, torch.Tensor)


def test_sfm_euler_sampler_adaptive_sigma():
    """SFM_Euler_sampler"""
    encoder = get_songunet()
    denoiser = SFMPrecondSR(
        img_resolution=[32, 32],
        img_in_channels=4,
        img_out_channels=6,
        N_grid_channels=4,
    ).to("cpu")
    dummy_net = torch.nn.Identity()

    batch_seeds = torch.as_tensor([1, 2]).to("cpu")
    image_rnd = torch.rand((2, 6, 32, 32)).to("cpu")
    rnd = StackedRandomGenerator(image_rnd.device, batch_seeds)

    cfg = {"rho": 7, "num_steps": 5, "sigma_min": 0.01, "t_min": 0.002}
    cfg = DictConfig(cfg)

    networks = {
        "encoder_net": dummy_net,
        "denoiser_net": denoiser,
    }

    out_val = SFM_Euler_sampler_Adaptive_Sigma(
        networks=networks, img_lr=image_rnd, randn_like=rnd.randn_like, cfg=cfg
    )
    assert isinstance(out_val, torch.Tensor)

    networks = {
        "encoder_net": encoder,
        "denoiser_net": denoiser,
    }

    out_val = SFM_Euler_sampler_Adaptive_Sigma(
        networks=networks, img_lr=image_rnd, randn_like=rnd.randn_like, cfg=cfg
    )
    assert isinstance(out_val, torch.Tensor)
