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

import os, time, psutil, hydra, torch, sys
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf
from torch.nn.parallel import DistributedDataParallel
from torch.utils.tensorboard import SummaryWriter
from modulus import Module
from modulus.models.diffusion import (
    SongUNet,
    EDMPrecondSR,
    SFMPrecondSR,
    SFMPrecondEmpty,
)
from modulus.distributed import DistributedManager
from modulus.launch.logging import PythonLogger, RankZeroLoggingWrapper
from modulus.metrics.diffusion import (
    RegressionLoss,
    ResLoss,
    SFMLoss,
    SFMLossSigmaPerChannel,
    SFMEncoderLoss,
)
from modulus.launch.logging import PythonLogger, RankZeroLoggingWrapper
from modulus.launch.utils import load_checkpoint, save_checkpoint
# Load utilities from corrdiff examples, make the corrdiff path absolute to avoid issues
sys.path.append(sys.path.append(os.path.join(os.path.dirname(__file__), "../corrdiff"))  )
from datasets.dataset import init_train_valid_datasets_from_config
from helpers.train_helpers import (
    set_patch_shape,
    set_seed,
    configure_cuda_for_consistent_precision,
    compute_num_accumulation_rounds,
    handle_and_clip_gradients,
    is_time_for_periodic_task,
)


# Train the CorrDiff model using the configurations in "conf/config_training.yaml"
@hydra.main(version_base="1.2", config_path="conf", config_name="config_training")
def main(cfg: DictConfig) -> None:
    # Initialize distributed environment for training
    DistributedManager.initialize()
    dist = DistributedManager()

    # Initialize loggers
    if dist.rank == 0:
        writer = SummaryWriter(log_dir="tensorboard")
    logger = PythonLogger("main")  # General python logger
    logger0 = RankZeroLoggingWrapper(logger, dist)  # Rank 0 logger

    # Resolve and parse configs
    OmegaConf.resolve(cfg)
    dataset_cfg = OmegaConf.to_container(cfg.dataset)
    if hasattr(cfg, "validation"):
        train_test_split = True
        validation_dataset_cfg = OmegaConf.to_container(cfg.validation)
    else:
        train_test_split = False
        validation_dataset_cfg = None

    fp_optimizations = cfg.training.perf.fp_optimizations
    fp16 = fp_optimizations == "fp16"
    enable_amp = fp_optimizations.startswith("amp")
    amp_dtype = torch.float16 if (fp_optimizations == "amp-fp16") else torch.bfloat16

    logger.info(f"Saving the outputs in {os.getcwd()}")
    checkpoint_dir = os.path.join(
        cfg.training.io.get("checkpoint_dir", "."), f"checkpoints_{cfg.model.name}"
    )
    if cfg.training.hp.batch_size_per_gpu == "auto":
        cfg.training.hp.batch_size_per_gpu = (
            cfg.training.hp.total_batch_size // dist.world_size
        )

    # Set seeds and configure CUDA and cuDNN settings to ensure consistent precision
    set_seed(dist.rank)
    configure_cuda_for_consistent_precision()

    # Instantiate the dataset
    data_loader_kwargs = {
        "pin_memory": True,
        "num_workers": cfg.training.perf.dataloader_workers,
        "prefetch_factor": cfg.training.perf.dataloader_workers,
    }
    (
        dataset,
        dataset_iterator,
        validation_dataset,
        validation_dataset_iterator,
    ) = init_train_valid_datasets_from_config(
        dataset_cfg,
        data_loader_kwargs,
        batch_size=cfg.training.hp.batch_size_per_gpu,
        seed=0,
        validation_dataset_cfg=validation_dataset_cfg,
        train_test_split=train_test_split,
    )

    # Parse image configuration & update model args
    dataset_channels = len(dataset.input_channels())
    img_in_channels = dataset_channels
    img_shape = dataset.image_shape()
    img_out_channels = len(dataset.output_channels())
    if cfg.model.hr_mean_conditioning:
        img_in_channels += img_out_channels
    patch_shape = (None, None)
    
    # Instantiate the model and move to device.
    if cfg.model.name not in (
        "regression", "sfm_encoder", "sfm", "sfm_two_stage",
    ):
        raise ValueError("Invalid model")
    model_args = {  # default parameters for all networks
        "img_out_channels": img_out_channels,
        "img_resolution": list(img_shape),
        "use_fp16": fp16,
    }
    standard_model_cfgs = {  # default parameters for different network types
        "regression": {
            "img_channels": 4,
            "N_grid_channels": 4,
            "embedding_type": "zero",
        },
        "sfm": {
            "img_channels": img_out_channels,
            "gridtype": "sinusoidal",
            "N_grid_channels": 4,
        }
    }

    model_args.update(standard_model_cfgs[cfg.model.name])
    if hasattr(cfg.model, "model_args"):  # override defaults from config file
        model_args.update(OmegaConf.to_container(cfg.model.model_args))
    
    if cfg.model.name == "regression":
        model = SongUNet(
            img_in_channels=img_in_channels + model_args["N_grid_channels"],
            **model_args,
        )
    elif cfg.model.name in "sfm_encoder":
        model = SFMPrecondEmpty()
    else: # sfm or sfm_two_stage
        model = SFMPrecondSR(
            img_in_channels=img_in_channels + model_args["N_grid_channels"],
            **model_args,
        )

    model.train().requires_grad_(True).to(dist.device)

    # Enable distributed data parallel if applicable
    if dist.world_size > 1:
        model = DistributedDataParallel(
            model,
            device_ids=[dist.local_rank],
            broadcast_buffers=True,
            output_device=dist.device,
            find_unused_parameters=dist.find_unused_parameters,
        )

    # Load the regression checkpoint if applicable
    if hasattr(cfg.training.io, "regression_checkpoint_path"):
        regression_checkpoint_path = to_absolute_path(
            cfg.training.io.regression_checkpoint_path
        )
        if not os.path.exists(regression_checkpoint_path):
            raise FileNotFoundError(
                f"Expected this regression checkpoint but not found: {regression_checkpoint_path}"
            )
        regression_net = Module.from_checkpoint(regression_checkpoint_path)
        regression_net.eval().requires_grad_(False).to(dist.device)
        logger0.success("Loaded the pre-trained regression model")

    # Instantiate the loss function
    if cfg.model.name == "diffusion":
        loss_fn = ResLoss(
            regression_net=regression_net,
            img_shape_x=img_shape[1],
            img_shape_y=img_shape[0],
            patch_shape_x=patch_shape[1],
            patch_shape_y=patch_shape[0],
            patch_num=patch_num,
            hr_mean_conditioning=cfg.model.hr_mean_conditioning,
        )
    elif cfg.model.name == "regression":
        loss_fn = RegressionLoss()
    elif cfg.model.name in ("sfm", "sfm_two_stage"):
        loss_fn = SFMLoss(
            encoder_loss_type = cfg.model.encoder_loss_type,
            encoder_loss_weight = cfg.model.encoder_loss_weight,
            sigma_min = cfg.model.sigma_min,
            sigma_data = 0.5,
        )
    elif cfg.model.name == "sfm_encoder":
        loss_fn = SFMEncoderLoss(
            encoder_loss_type = cfg.model.encoder_loss_type,
        )
    else:
        raise NotImplementedError(f"Model {cfg.model.name} not supported.")

    # Instantiate the optimizer
    if encoder_net is not None and cfg.task != "sfm_two_stage":
        params = list(denoiser_net.parameters()) + list(encoder_net.parameters())
    else:
        params = model.parameters()

    optimizer = torch.optim.Adam(
        params=params, lr=cfg.training.hp.lr, betas=[0.9, 0.999], eps=1e-8
    )

    # Record the current time to measure the duration of subsequent operations.
    start_time = time.time()

    # Compute the number of required gradient accumulation rounds
    # It is automatically used if batch_size_per_gpu * dist.world_size < total_batch_size
    batch_gpu_total, num_accumulation_rounds = compute_num_accumulation_rounds(
        cfg.training.hp.total_batch_size,
        cfg.training.hp.batch_size_per_gpu,
        dist.world_size,
    )
    batch_size_per_gpu = cfg.training.hp.batch_size_per_gpu
    logger0.info(f"Using {num_accumulation_rounds} gradient accumulation rounds")

    ## Resume training from previous checkpoints if exists
    ### TODO needs to be redone, need to store model + encoder + optimizer 
    if dist.world_size > 1:
        torch.distributed.barrier()
    try:
        cur_nimg = load_checkpoint(
            path=checkpoint_dir,
            models=[sfm_encoder, denoiser],
            optimizer=optimizer,
            device=dist.device,
        )
    except:
        cur_nimg = 0

    ############################################################################
    #                            MAIN TRAINING LOOP                            #
    ############################################################################

if __name__ == "__main__":
    main()
