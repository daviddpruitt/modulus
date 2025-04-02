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

from typing import List, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from physicsnemo.models.diffusion import SongUNetPosEmbd


class SFMLoss:
    """
    Loss function corresponding to Stochastic Flow matching

    Parameters
    ----------
    encoder_loss_type: str
        Type of loss to use ["l1", "l2", None]
    encoder_loss_weight: float
        Regularizer loss weights, by defaults 0.1.
    sigma_min: Union[List[float], float]
        Minimum value of noise sigma, default 2e-3
        Protects against values near zero that result in loss explosion.
    sigma_data: float
        EDM weighting, default 0.5
    """

    def __init__(
        self,
        encoder_loss_type: str = "l2",
        encoder_loss_weight: float = 0.1,
        sigma_min: Union[List[float], float] = 0.002,
        sigma_data: float = 0.5,
    ):
        """
        Loss function corresponding to Stochastic Flow matching

        Parameters
        ----------
        encoder_loss_type: str, optional
            Type of loss to use ["l1", "l2", None], defaults to 'l2'
        encoder_loss_weight: float, optional
            Regularizer loss weights, by defaults 0.1.
        sigma_min: Union[List[float], float], optional
            Minimum value of noise sigma, default 2e-3
            Protects against values near zero that result in loss explosion.
        sigma_data: float, optional
            EDM weighting, default 0.5
        """
        self.encoder_loss_type = encoder_loss_type
        self.encoder_loss_weight = encoder_loss_weight
        self.sigma_min = sigma_min
        self.sigma_data = sigma_data

        if encoder_loss_type not in ["l1", "l2", None]:
            raise ValueError(
                f"encoder_loss_type should be one of ['l1', 'l2', None] not {encoder_loss_type}"
            )

    def __call__(
        self,
        denoiser_net: torch.nn.Module,
        encoder_net: torch.nn.Module,
        img_clean: torch.Tensor,
        img_lr: torch.Tensor,
    ):
        """
        Calculate the loss for corresponding to stochastic flow matching

        Parameters
        ----------
            denoiser_net: torch.Tensor
                The denoiser network making the predictions
            encoder_net: torch.Tensor
                The encoder network making the predictions
            img_clean: torch.Tensor
                Input images (high resolution) to the neural network.
            img_lr: torch.Tensor
                Input images (low resolution) to the neural network.
        Returns
        -------
            torch.Tensor
            A tensor representing the combined loss calculated based on the flow matching
            encoder and denoiser networks
        """
        # uniformly samples from 0 to 1 in torch
        if isinstance(denoiser_net, torch.nn.parallel.DistributedDataParallel):
            sigma_max_per_channel = denoiser_net.module.get_sigma_max().to(
                device=img_clean.device
            )
        else:
            sigma_max_per_channel = denoiser_net.get_sigma_max().to(
                device=img_clean.device
            )

        # clamp to min value
        if len(self.sigma_min) > 1:
            sigma_max_per_channel = torch.maximum(
                sigma_max_per_channel,
                torch.tensor(self.sigma_min, device=img_clean.device),
            )
            # Normalize from 0 to 1
            sigma_max = 1.0
        else:
            # just use the first value, ignore the rest
            sigma_max = torch.maximum(
                sigma_max_per_channel,
                torch.tensor(self.sigma_min, device=img_clean.device),
            )[0]

        rnd_uniform = torch.rand([img_clean.shape[0], 1, 1, 1], device=img_clean.device)

        sampled_sigma = rnd_uniform * sigma_max
        weight = (sampled_sigma**2 + self.sigma_data**2) / (
            sampled_sigma * self.sigma_data
        ) ** 2

        # augment for conditional generaiton
        x_tot = torch.cat((img_clean, img_lr), dim=1)

        x_1 = x_tot[:, : img_clean.shape[1], :, :]  # x_1 - target
        x_low = x_tot[:, img_clean.shape[1] :, :, :]  # x_low - upsampled ERA5

        # encode the low resolution data x_low to x_0
        # check same if encoder_net is in distributed data parallel

        if isinstance(encoder_net, SongUNetPosEmbd) or (
            isinstance(encoder_net, nn.parallel.DistributedDataParallel)
            and isinstance(encoder_net.module, SongUNetPosEmbd)
        ):
            x_0 = encoder_net(x_low, noise_labels=torch.tensor([0]), class_labels=None)
        else:
            x_0 = encoder_net(x_low)

        # convert sigma to time
        # sampled_sigma = (1-t)*sigma_max
        time = 1 - sampled_sigma / sigma_max  # this is the time from 1 to 0

        # we don't subtract x_0, this will be done in the sampler
        x_t = ((1 - time) * x_0) + (time * x_1)
        if len(self.sigma_min) > 1:
            sigma_t = (
                sigma_max_per_channel.unsqueeze(0).unsqueeze(2).unsqueeze(2)
                * sampled_sigma
            )
        else:
            sigma_t = sampled_sigma
        x_t_noisy = x_t + torch.randn_like(x_0) * sigma_t

        D_x_t = denoiser_net(
            x=x_t_noisy,
            sigma=sampled_sigma,
            condition=x_low,
        )
        # time_weight = lambda t: 1/(1 - torch.clamp(t, 0.9))
        # time_weight = lambda t: 1
        def time_weight(t):
            return 1

        sfm_loss = weight * ((time_weight(time) * (D_x_t - x_1)) ** 2)

        if self.encoder_loss_type == "l1":
            encoder_loss = F.l1_loss(x_1, x_0, reduction="none")
            weighted_encoder_loss = self.encoder_loss_weight * encoder_loss
        elif self.encoder_loss_type == "l2":
            encoder_loss = F.mse_loss(x_1, x_0, reduction="none")
            weighted_encoder_loss = self.encoder_loss_weight * encoder_loss
        elif self.encoder_loss_type is None:
            encoder_loss = torch.tensor(
                0.0
            )  # This covers the case where there is no encoder_loss
            weighted_encoder_loss = torch.tensor(0.0)

        return sfm_loss + weighted_encoder_loss


class SFMEncoderLoss:
    """
    Loss function corresponding to Stochastic Flow matching for the encoder portion

    Parameters
    ----------
    encoder_loss_type: str
        Type of loss to use ["l1", "l2", None]
    """

    def __init__(self, encoder_loss_type: str, **kwargs):
        if encoder_loss_type not in ["l1", "l2"]:
            raise ValueError(
                f"encoder_loss_type should be either l1 or l2 not {encoder_loss_type}"
            )
        self.encoder_loss_type = encoder_loss_type

    def __call__(
        self,
        denoiser_net: torch.nn.Module,
        encoder_net: torch.nn.Module,
        img_clean: torch.Tensor,
        img_lr: torch.Tensor,
    ):
        """
        Calculate the loss for the enoder used in stochastic flow matching

        Parameters
        ----------
        models: [torch.Tensor, torch.Tensor]
            The denoiser and encoder networks making the predictions
            Stored as [denoiser, encoder]
        img_clean: torch.Tensor
            Input images (high resolution) to the neural network.
        img_lr: torch.Tensor
            Input images (low resolution) to the neural network.

        Returns
        -------
            torch.Tensor
            A tensor representing the loss calculated based on the encoder's
            predictions
        """
        x_1 = img_clean
        x_low = img_lr

        if isinstance(encoder_net, SongUNetPosEmbd) or (
            isinstance(encoder_net, nn.parallel.DistributedDataParallel)
            and isinstance(encoder_net.module, SongUNetPosEmbd)
        ):
            x_0 = encoder_net(x_low, noise_labels=torch.tensor([0]), class_labels=None)
        else:
            x_0 = encoder_net(x_low)

        if self.encoder_loss_type == "l1":
            encoder_loss = F.l1_loss(x_1, x_0, reduction="none")
        elif self.encoder_loss_type == "l2":
            encoder_loss = F.mse_loss(x_1, x_0, reduction="none")

        return encoder_loss
