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

from modulus.models.diffusion import SongUNetPosEmbd
from modulus.models.diffusion import Conv2dSerializable
import torch


def get_encoder(cfg):
    in_channels = len(cfg.dataset["in_channels"])
    out_channels = len(cfg.dataset["out_channels"])
    encoder_type = cfg.model["encoder_type"]

    if encoder_type == "1x1conv":
        encoder = Conv2dSerializable(in_channels, out_channels, kernel_size=1)
    elif "songunet" in encoder_type:
        model_channels_dict = {
            "songunet_s": 32,  # 11.60M
            "songunet_xs": 16,  #  2.90M
            "songunet_2xs": 8,  #  0.74M
            "songunet_3xs": 4,  #  0.19M
        }
        if hasattr(cfg.model, "songunet_checkpoint_level"):
            songunet_checkpoint_level = cfg.model.songunet_checkpoint_level
        else:
            songunet_checkpoint_level = None

        songunet_kwargs = {
            "embedding_type": "zero",
            "label_dim": 0,
            "encoder_type": "standard",
            "decoder_type": "standard",
            "channel_mult_noise": 1,
            "resample_filter": [1, 1],
            "channel_mult": [1, 2, 2, 4, 4],
            "attn_resolutions": [28],
            "N_grid_channels": 0,
            "dropout": cfg.model.dropout,
            "checkpoint_level": songunet_checkpoint_level,
            "model_channels": model_channels_dict[encoder_type],
        }
        encoder = SongUNetPosEmbd(
            img_resolution=cfg.dataset["img_shape_x"],
            in_channels=in_channels,
            out_channels=out_channels,
            **songunet_kwargs,
        )
    else:
        raise ValueError(f"Unknown encoder type: {encoder_type}")

    return encoder
