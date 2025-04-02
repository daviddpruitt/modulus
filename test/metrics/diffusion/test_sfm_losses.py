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

import re

import pytest
import torch

from physicsnemo.metrics.diffusion import (
    SFMEncoderLoss,
    SFMLoss,
)
from physicsnemo.models.diffusion import SongUNetPosEmbd


class fake_net:
    """dummy class to test sfm encoder"""

    def get_sigma_max():
        return torch.tensor([1.0])


def get_songunet():
    """helper that creates a songunet for testing"""
    songunet_kwargs = {
        "img_resolution": 64,
        "in_channels": 4,
        "out_channels": 4,
        "embedding_type": "zero",
        "label_dim": 0,
        "encoder_type": "standard",
        "decoder_type": "standard",
        "channel_mult_noise": 1,
        "resample_filter": [1, 1],
        "channel_mult": [1, 2, 2, 4, 4],
        "attn_resolutions": [28],
        "N_grid_channels": 0,
        "model_channels": 4,
    }
    return SongUNetPosEmbd(**songunet_kwargs)


def test_sfmloss_initialization():
    loss_fn = SFMLoss()

    assert loss_fn.encoder_loss_type == "l2"
    assert loss_fn.encoder_loss_weight == 0.1
    assert loss_fn.sigma_min == 0.002
    assert loss_fn.sigma_data == 0.5

    loss_fn = SFMLoss(
        encoder_loss_type="l1",
        encoder_loss_weight=[0.1, 0.2],
        sigma_min=5e-4,
        sigma_data=0.1,
    )

    assert loss_fn.encoder_loss_type == "l1"
    assert loss_fn.encoder_loss_weight == [0.1, 0.2]
    assert loss_fn.sigma_min == 5e-4
    assert loss_fn.sigma_data == 0.1

    # test for invalid loss type
    with pytest.raises(
        ValueError,
        match=re.escape(
            "encoder_loss_type should be one of ['l1', 'l2', None] not bogus"
        ),
    ):
        loss_fn = SFMLoss(
            encoder_loss_type="bogus",
        )


def test_sfmloss_call():
    # dummy network for loss
    dummy_denoiser = fake_net()
    dummy_encoder = get_songunet()
    dummy_net = torch.nn.Identity()

    image_zeros = torch.zeros((2, 2))

    # test defaults, encoder l2 loss, sigma_min is float
    loss_fn = SFMLoss()

    loss_value = loss_fn(dummy_denoiser, dummy_net, image_zeros, image_zeros)
    assert isinstance(loss_value, torch.Tensor)

    loss_value = loss_fn(dummy_denoiser, dummy_net, image_zeros, image_zeros)
    assert isinstance(loss_value, torch.Tensor)

    # test encoder l1 loss, sigma_min is list
    loss_fn = SFMLoss(encoder_loss_type="l1", sigma_min=[0.001, 0.001])
    loss_value = loss_fn(dummy_denoiser, dummy_net, image_zeros, image_zeros)
    assert isinstance(loss_value, torch.Tensor)

    # test no encoder loss, sigma_min is list
    loss_fn = SFMLoss(encoder_loss_type=None)
    loss_value = loss_fn(dummy_denoiser, dummy_net, image_zeros, image_zeros)
    assert isinstance(loss_value, torch.Tensor)

    # test with ddp encoder and SongUnetPosEmbed denoiser
    dummy_ddp_denoiser = torch.nn.parallel.DistributedDataParallel(dummy_denoiser)
    loss_value = loss_fn(dummy_ddp_denoiser, dummy_encoder, image_zeros, image_zeros)
    assert isinstance(loss_value, torch.Tensor)


def test_sfmencoderloss_initialization():
    loss_fn = SFMEncoderLoss()

    assert loss_fn.encoder_loss_type == "l2"

    # test for invalid loss type
    with pytest.raises(
        ValueError,
        match="encoder_loss_type should be either l1 or l2 not bogus",
    ):
        loss_fn = SFMEncoderLoss(
            encoder_loss_type="bogus",
        )


def test_sfmencoderloss_call():
    # dummy network for loss
    dummy_net = torch.nn.Identity()

    image_zeros = torch.zeros((2, 2))
    image_twos = torch.ones((2, 2)) * 2

    # test l1 loss
    loss_fn = SFMEncoderLoss()

    # encoder loss is deterministic
    loss_value = loss_fn(dummy_net, dummy_net, image_twos, image_twos)
    assert torch.equal(loss_value, image_zeros)

    loss_value = loss_fn(dummy_net, dummy_net, image_zeros, image_twos)
    assert torch.equal(loss_value, image_twos * image_twos)

    # test l2 loss
    loss_fn = SFMEncoderLoss("l1")

    # encoder loss is deterministic
    loss_value = loss_fn(dummy_net, dummy_net, image_twos, image_twos)
    assert torch.equal(loss_value, image_zeros)

    loss_value = loss_fn(dummy_net, dummy_net, image_zeros, image_twos)
    assert torch.equal(loss_value, image_twos)
