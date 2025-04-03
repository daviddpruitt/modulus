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

from physicsnemo.models.diffusion import Conv2dSerializable


def test_conv2dserializable_initialization():
    conv_net = Conv2dSerializable(in_channels=8, out_channels=4, kernel_size=3)

    assert conv_net.in_channels == 8
    assert conv_net.out_channels == 4
    assert conv_net.kernel_size == 3


def test_conv2dserializable_forward():
    test_data = torch.zeros(1, 8, 16, 16)
    conv_net = Conv2dSerializable(in_channels=8, out_channels=4, kernel_size=3)

    out_tensor = conv_net(test_data)
    assert out_tensor.shape == (1, 4, 16, 16)
