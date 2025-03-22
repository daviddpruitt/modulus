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

import copy
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
    #SFMLossSigmaPerChannel,
    SFMEncoderLoss,
)
from modulus.launch.logging import PythonLogger, RankZeroLoggingWrapper
from modulus.launch.utils import load_checkpoint, save_checkpoint
# Load utilities from corrdiff examples, make the corrdiff path absolute to avoid issues
#sys.path.append(sys.path.append(os.path.join(os.path.dirname(__file__), "../corrdiff"))  )
from datasets.dataset import init_train_valid_datasets_from_config
from helpers.train_helpers import (
    set_patch_shape,
    set_seed,
    configure_cuda_for_consistent_precision,
    compute_num_accumulation_rounds,
    handle_and_clip_gradients,
    is_time_for_periodic_task,
)
from helpers.sfm_utils import get_encoder


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
    patch_shape = (None, None)
    
    # Instantiate the model and move to device.
    if cfg.model.name not in (
        "sfm_encoder", "sfm", "sfm_two_stage",
    ):
        raise ValueError("Invalid model")
    model_args = {  # default parameters for all networks
        "img_out_channels": img_out_channels,
        "img_resolution": list(img_shape),
        "use_fp16": fp16,
    }
    ## remaining defaults are what we want
    standard_model_cfgs = {  # default parameters for different network types
        "sfm": {
            "gridtype": "sinusoidal",
            "N_grid_channels": 4,
        },
        "sfm_two_stage": {
            "gridtype": "sinusoidal",
            "N_grid_channels": 4,
        },
        "sfm_encoder": {}, # empty preconditioner
    }

    model_args.update(standard_model_cfgs[cfg.model.name])
    if hasattr(cfg.model, "model_args"):  # override defaults from config file
        model_args.update(OmegaConf.to_container(cfg.model.model_args))
    
    if cfg.model.name == "sfm_encoder":
        # should this be set to no_grad?
        denoiser_net = SFMPrecondEmpty()
    else: # sfm or sfm_two_stage
        denoiser_net = SFMPrecondSR(
            img_in_channels=img_in_channels + model_args["N_grid_channels"],
            **model_args,
        )

    denoiser_net.train().requires_grad_(True).to(dist.device)
    denoiser_ema = copy.deepcopy(denoiser_net).eval().requires_grad_(False)
    ema_halflife_nimg = int(cfg.training.hp.ema * 1000000)
    if hasattr(cfg.training.hp, "ema_rampup_ratio"):
        ema_rampup_ratio = float(cfg.training.hp.ema_rampup_ratio)
    else:
        ema_rampup_ratio = 0.5

    # Create or load the encoder:
    if cfg.model.name in ["sfm", "sfm_encoder"]:
        encoder_net = get_encoder(cfg)
        encoder_net.train().requires_grad_(True).to(dist.device)
        logger0.success("Constructed encoder network succesfully")
    else: # "sfm_two_stage"
        if not hasattr(cfg.training.io, "encoder_checkpoint_path"):
            raise KeyError("Need to provide encoder_checkpoint_path when using sfm_two_stage")
        encoder_checkpoint_path = to_absolute_path(
            cfg.training.io.encoder_checkpoint_path
        )
        if not os.path.exists(encoder_checkpoint_path):
            raise FileNotFoundError(
                f"Expected this encoder checkpoint but not found: {encoder_checkpoint_path}"
            )
        encoder_net = Module.from_checkpoint(encoder_checkpoint_path)
        encoder_net.eval().requires_grad_(False).to(device)
        logger0.success("Loaded the pre-trained encoder network")


    # Instantiate the loss function(s)
    if cfg.model.name in ("sfm", "sfm_two_stage"):
        loss_fn = SFMLoss(
            encoder_loss_type = cfg.model.encoder_loss_type,
            encoder_loss_weight = cfg.model.encoder_loss_weight,
            sigma_min = cfg.model.sigma_min,
        )
        # with sfm the encoder and diffusion model are trained together
        if cfg.model.name == "sfm":
            loss_fn_encoder = SFMEncoderLoss(encoder_loss_type='l2')
    elif cfg.model.name == "sfm_encoder":
        loss_fn = SFMEncoderLoss(
            encoder_loss_type = cfg.model.encoder_loss_type,
        )
    else:
        raise NotImplementedError(f"Model {cfg.model.name} not supported.")

    # Instantiate the optimizer
    if cfg.model.name == "sfm_two_stage":
        params = denoiser_net.parameters()
    else:
        params = list(denoiser_net.parameters()) + list(encoder_net.parameters())

    optimizer = torch.optim.Adam(
        params=params, lr=cfg.training.hp.lr, betas=[0.9, 0.999], eps=1e-8
    )

    # Enable distributed data parallel if applicable
    if dist.world_size > 1:
        denoiser_net = DistributedDataParallel(
            denoiser_net,
            device_ids=[dist.local_rank],
            broadcast_buffers=True,
            output_device=dist.device,
            find_unused_parameters=dist.find_unused_parameters,
        )
        if cfg.model.name != "sfm_two_stage":
            encoder_net = DistributedDataParallel(
                encoder_net,
                device_ids=[dist.local_rank],
                broadcast_buffers=True,
                output_device=dist.device,
                find_unused_parameters=dist.find_unused_parameters,
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

    logger0.info(f"Training for {cfg.training.hp.training_duration} images...")
    done = False

    # init variables to monitor running mean of average loss since last periodic
    average_loss_running_mean = 0
    n_average_loss_running_mean = 1

    while not done:
        tick_start_nimg = cur_nimg
        tick_start_time = time.time()
        # Compute & accumulate gradients
        optimizer.zero_grad(set_to_none=True)
        loss_accum = 0
        for _ in range(num_accumulation_rounds):
            img_clean, img_lr, labels = next(dataset_iterator)
            img_clean = img_clean.to(dist.device).to(torch.float32).contiguous()
            img_lr = img_lr.to(dist.device).to(torch.float32).contiguous()
            labels = labels.to(dist.device).contiguous()
            with torch.autocast("cuda", dtype=amp_dtype, enabled=enable_amp):
                loss = loss_fn(
                    denoiser_net=denoiser_net,
                    encoder_net=encoder_net,
                    img_clean=img_clean,
                    img_lr=img_lr,
                    labels=labels,
                    augment_pipe=None,
                )
            loss = loss.sum() / batch_size_per_gpu
            loss_accum += loss / num_accumulation_rounds
            loss.backward()

        loss_sum = torch.tensor([loss_accum], device=dist.device)
        if dist.world_size > 1:
            torch.distributed.barrier()
            torch.distributed.all_reduce(loss_sum, op=torch.distributed.ReduceOp.SUM)
        average_loss = (loss_sum / dist.world_size).cpu().item()

        # update running mean of average loss since last periodic task
        average_loss_running_mean += (
            average_loss - average_loss_running_mean
        ) / n_average_loss_running_mean
        n_average_loss_running_mean += 1

        if dist.rank == 0:
            writer.add_scalar("training_loss", average_loss, cur_nimg)
            writer.add_scalar(
                "training_loss_running_mean", average_loss_running_mean, cur_nimg
            )

        ptt = is_time_for_periodic_task(
            cur_nimg,
            cfg.training.io.print_progress_freq,
            done,
            cfg.training.hp.total_batch_size,
            dist.rank,
            rank_0_only=True,
        )
        if ptt:
            # reset running mean of average loss
            average_loss_running_mean = 0
            n_average_loss_running_mean = 1

        # Update weights.
        lr_rampup = cfg.training.hp.lr_rampup  # ramp up the learning rate
        for g in optimizer.param_groups:
            if lr_rampup > 0:
                g["lr"] = cfg.training.hp.lr * min(cur_nimg / lr_rampup, 1)
            if cur_nimg >= lr_rampup:
                g["lr"] *= cfg.training.hp.lr_decay ** ((cur_nimg - lr_rampup) // 5e6)
            current_lr = g["lr"]
            if dist.rank == 0:
                writer.add_scalar("learning_rate", current_lr, cur_nimg)

        # clear any nans from the denoiser
        for param in denoiser_net.parameters():
            if param.grad is not None:
                torch.nan_to_num(
                    param.grad, nan=0, posinf=1e5, neginf=-1e5, out=param.grad
                )
        handle_and_clip_gradients(
            denoiser_net, grad_clip_threshold=cfg.training.hp.grad_clip_threshold
        )
        handle_and_clip_gradients(
            encoder_net, grad_clip_threshold=cfg.training.hp.grad_clip_threshold
        )
        optimizer.step()

        # Update EMA.
        if ema_rampup_ratio is not None:
            ema_halflife_nimg = min(ema_halflife_nimg, cur_nimg * ema_rampup_ratio)
        ema_beta = 0.5 ** (cfg.training.hp.total_batch_size / max(ema_halflife_nimg, 1e-8))
        for p_ema, p_net in zip(denoiser_ema.parameters(), denoiser_net.parameters()):
            p_ema.copy_(p_net.detach().lerp(p_ema, ema_beta))

        cur_nimg += cfg.training.hp.total_batch_size
        done = cur_nimg >= cfg.training.hp.training_duration

        # Validation
        if validation_dataset_iterator is not None:
            valid_loss_accum = 0
            if is_time_for_periodic_task(
                cur_nimg,
                cfg.training.io.validation_freq,
                done,
                cfg.training.hp.total_batch_size,
                dist.rank,
            ):
                rmse_encoder_valid_accum_mean = 0
                with torch.no_grad():
                    for _ in range(cfg.training.io.validation_steps):
                        img_clean_valid, img_lr_valid, labels_valid = next(
                            validation_dataset_iterator
                        )

                        img_clean_valid = (
                            img_clean_valid.to(dist.device)
                            .to(torch.float32)
                            .contiguous()
                        )
                        img_lr_valid = (
                            img_lr_valid.to(dist.device).to(torch.float32).contiguous()
                        )
                        labels_valid = labels_valid.to(dist.device).contiguous()
                        loss_valid = loss_fn(
                            denoiser_net=denoiser_net,
                            encoder_net=encoder_net,
                            img_clean=img_clean_valid,
                            img_lr=img_lr_valid,
                            labels=labels_valid,
                            augment_pipe=None,
                        )
                        loss_valid = (
                            (loss_valid.sum() / batch_size_per_gpu).cpu().item()
                        )
                        valid_loss_accum += (
                            loss_valid / cfg.training.io.validation_steps
                        )

                        if cfg.model.name == "sfm":
                            rmse_encoder_valid = loss_fn_encoder(
                                denoiser_net=denoiser_net,
                                encoder_net=encoder_net,
                                img_clean=img_clean_valid,
                                img_lr=img_lr_valid,
                                labels=labels_valid,
                                augment_pipe=augment_pipe,
                                loggers=None
                            )
                            rmse_encoder_valid_accum_mean += rmse_encoder_valid.mean((0,2,3)) / cfg.validation_steps

                    valid_loss_sum = torch.tensor(
                        [valid_loss_accum], device=dist.device
                    )
                    if dist.world_size > 1:
                        torch.distributed.barrier()
                        torch.distributed.all_reduce(
                            valid_loss_sum, op=torch.distributed.ReduceOp.SUM
                        )
                    average_valid_loss = valid_loss_sum / dist.world_size
                    if dist.rank == 0:
                        writer.add_scalar(
                            "validation_loss", average_valid_loss, cur_nimg
                        )

                if dist.rank == 0:
                    if cfg.model.name == "sfm" and cfg.model.sfm['sigma_max']['learnable']:
                        denoiser_net.update_sigma_max(rmse_encoder_valid_accum_mean)


        if is_time_for_periodic_task(
            cur_nimg,
            cfg.training.io.print_progress_freq,
            done,
            cfg.training.hp.total_batch_size,
            dist.rank,
            rank_0_only=True,
        ):
            # Print stats if we crossed the printing threshold with this batch
            tick_end_time = time.time()
            fields = []
            fields += [f"samples {cur_nimg:<9.1f}"]
            fields += [f"training_loss {average_loss:<7.2f}"]
            fields += [f"training_loss_running_mean {average_loss_running_mean:<7.2f}"]
            fields += [f"learning_rate {current_lr:<7.8f}"]
            fields += [f"total_sec {(tick_end_time - start_time):<7.1f}"]
            fields += [f"sec_per_tick {(tick_end_time - tick_start_time):<7.1f}"]
            fields += [
                f"sec_per_sample {((tick_end_time - tick_start_time) / (cur_nimg - tick_start_nimg)):<7.2f}"
            ]
            fields += [
                f"cpu_mem_gb {(psutil.Process(os.getpid()).memory_info().rss / 2**30):<6.2f}"
            ]
            fields += [
                f"peak_gpu_mem_gb {(torch.cuda.max_memory_allocated(dist.device) / 2**30):<6.2f}"
            ]
            fields += [
                f"peak_gpu_mem_reserved_gb {(torch.cuda.max_memory_reserved(dist.device) / 2**30):<6.2f}"
            ]
            logger0.info(" ".join(fields))
            torch.cuda.reset_peak_memory_stats()

        # Save checkpoints
        if dist.world_size > 1:
            torch.distributed.barrier()
        if is_time_for_periodic_task(
            cur_nimg,
            cfg.training.io.save_checkpoint_freq,
            done,
            cfg.training.hp.total_batch_size,
            dist.rank,
            rank_0_only=True,
        ):
            save_checkpoint(
                path=checkpoint_dir,
                models=[denoiser_net, encoder_net],
                optimizer=optimizer,
                epoch=cur_nimg,
            )

    # Done.
    logger0.info("Training Completed.")
    

if __name__ == "__main__":
    main()
