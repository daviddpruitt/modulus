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
    precond = SFMPrecondEmpty()

    assert isinstance(precond, SFMPrecondEmpty)
    assert precond.label_dim is None
    assert precond.param == torch.tensor(0.0)


# SFMPrecondSR tests
def test_sfmprecondsr_initialization():
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
    )

    assert isinstance(precond, SFMPrecondSR)
    assert precond.img_shape_y == 32
    assert precond.img_shape_x == 32
    assert torch.equal(precond.sigma_max_current, torch.Tensor([0.5, 0.5, 0.5]))
    assert torch.equal(precond.ema_weight, torch.tensor(0.2))
    assert torch.equal(precond.min_values, torch.tensor(0.1))
    assert precond.sigma_data == 0.3
