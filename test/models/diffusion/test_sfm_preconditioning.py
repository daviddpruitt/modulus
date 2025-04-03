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

from physicsnemo.models.diffusion import SFMPrecondEmpty, SFMPrecondSR


# SFMPrecondEmpty test
def test_sfmprecondempty_initialzation():
    """checks SFMPrecondEmpty __init__"""
    precond = SFMPrecondEmpty()

    assert isinstance(precond, SFMPrecondEmpty)
    assert precond.label_dim is None
    assert precond.param == torch.tensor(0.0)


# SFMPrecondSR tests
def test_sfmprecondsr_initialization():
    """checks SFMPrecondSR __init__"""
    precond = SFMPrecondSR(
        img_resolution=[32, 32],
        img_in_channels=4,
        img_out_channels=6,
        sigma_data=0.3,
    )

    assert isinstance(precond, SFMPrecondSR)
    assert precond.img_shape_y == 32
    assert precond.img_shape_x == 32
    assert precond.sigma_data == 0.3

    # test with sigma_max as a dict
    sigma_max = {
        "initial_values": [0.5, 0.5, 0.5],
        "ema_weight": 0.2,
        "min_values": 0.1,
    }

    precond = SFMPrecondSR(
        img_resolution=[32, 32],
        img_in_channels=4,
        img_out_channels=6,
        sigma_max=sigma_max,
        sigma_data=0.3,
        encoder_type="l1",
    )

    assert isinstance(precond, SFMPrecondSR)
    assert precond.img_shape_y == 32
    assert precond.img_shape_x == 32
    assert torch.equal(precond.sigma_max_current, torch.Tensor([0.5, 0.5, 0.5]))
    assert torch.equal(precond.ema_weight, torch.tensor(0.2))
    assert torch.equal(precond.min_values, torch.tensor(0.1))
    assert precond.sigma_data == 0.3


# SFMPrecondSR tests
def test_sfmprecondsr_sigma():
    """checks update_sigma_max and get_sigma_max"""

    precond = SFMPrecondSR(
        img_resolution=[32, 32],
        img_in_channels=4,
        img_out_channels=6,
        sigma_data=0.3,
        sigma_max=8.0,
    )

    assert precond.get_sigma_max() == 8.0

    precond.update_sigma_max(16.0)
    assert precond.sigma_max_current == 16.0

    # test with ema as a dict
    ema_weight = 0.5
    init_vals = torch.tensor([16.0, 1.0, 0.5])
    updated_vals = torch.tensor([8.0, 2.0, 2.0])
    expected_vals = init_vals * 0.5 + updated_vals * 0.5

    sigma_max = {
        "initial_values": init_vals.tolist(),
        "ema_weight": ema_weight,
        "min_values": 0.1,
    }

    precond = SFMPrecondSR(
        img_resolution=[32, 32],
        img_in_channels=4,
        img_out_channels=6,
        sigma_max=sigma_max,
        sigma_data=0.3,
        encoder_type="l1",
    )

    assert torch.equal(precond.sigma_max_current, init_vals)

    precond.update_sigma_max(updated_vals.tolist())
    assert torch.equal(precond.sigma_max_current, expected_vals)

    assert torch.equal(precond.round_sigma(init_vals.tolist()), init_vals)


def test_sfmprecondsr_forward():
    """checks forward"""

    image_in = torch.zeros((2, 4, 32, 32))
    image_cond = torch.zeros((2, 6, 32, 32))
    sigma = torch.rand((2, 1, 1, 1))

    precond = SFMPrecondSR(
        img_resolution=[32, 32],
        img_in_channels=6,
        img_out_channels=4,
        N_grid_channels=4,
    )

    out_val = precond(image_in, sigma=sigma, condition=None)
    assert isinstance(out_val, torch.Tensor)

    precond = SFMPrecondSR(
        img_resolution=[32, 32],
        img_in_channels=6,
        img_out_channels=4,
        N_grid_channels=4,
        use_x_low_conditioning=True,
    )

    out_val = precond(image_in, sigma=sigma, condition=image_cond)
    assert isinstance(out_val, torch.Tensor)
