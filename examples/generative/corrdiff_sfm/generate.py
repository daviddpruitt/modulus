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

import hydra
from omegaconf import OmegaConf, DictConfig
import torch
import torch._dynamo
import nvtx
import numpy as np
import netCDF4 as nc
from physicsnemo.distributed import DistributedManager
from physicsnemo.launch.logging import PythonLogger, RankZeroLoggingWrapper
from physicsnemo import Module
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from einops import rearrange
from torch.distributed import gather
import tqdm


from hydra.utils import to_absolute_path
from physicsnemo.utils.generative import SFM_Euler_sampler, SFM_Euler_sampler_Adaptive_Sigma, StackedRandomGenerator, SFM_encoder_sampler
from physicsnemo.utils.corrdiff import (
    NetCDFWriter,
    get_time_from_range,
)


from helpers.generate_helpers import (
    get_dataset_and_sampler,
    save_images,
)
from helpers.train_helpers import set_patch_shape


@hydra.main(version_base="1.2", config_path="conf", config_name="config_generate")
def main(cfg: DictConfig) -> None:
    """Generate random images using the techniques described in the paper
    "Elucidating the Design Space of Diffusion-Based Generative Models".
    """

    # Initialize distributed manager
    DistributedManager.initialize()
    dist = DistributedManager()
    device = dist.device

    # Initialize logger
    logger = PythonLogger("generate")  # General python logger
    logger0 = RankZeroLoggingWrapper(logger, dist)
    logger.file_logging("generate.log")

    # Handle the batch size
    seeds = list(np.arange(cfg.generation.num_ensembles))
    num_batches = (
        (len(seeds) - 1) // (cfg.generation.seed_batch_size * dist.world_size) + 1
    ) * dist.world_size
    all_batches = torch.as_tensor(seeds).tensor_split(num_batches)
    rank_batches = all_batches[dist.rank :: dist.world_size]

    # Synchronize
    if dist.world_size > 1:
        torch.distributed.barrier()

    # Parse the inference input times
    if cfg.generation.times_range and cfg.generation.times:
        raise ValueError("Either times_range or times must be provided, but not both")
    if cfg.generation.times_range:
        times = get_time_from_range(cfg.generation.times_range)
    else:
        times = cfg.generation.times

    # Create dataset object
    dataset_cfg = OmegaConf.to_container(cfg.dataset)
    dataset, sampler = get_dataset_and_sampler(dataset_cfg=dataset_cfg, times=times)
    img_shape = dataset.image_shape()
    img_out_channels = len(dataset.output_channels())

    # patching not supported for 
    patch_shape = (None, None)
    img_shape, patch_shape = set_patch_shape(img_shape, patch_shape)

    # Parse the inference mode
    if cfg.generation.inference_mode not in  ["sfm", "sfm_encoder", "sfm_two_stage"]:
        raise ValueError(f"Invalid inference mode {cfg.generation.inference_mode}")

    # Load networks, move to device, change precision
    encoder_ckpt_filename = cfg.generation.io.encoder_ckpt_filename
    logger0.info(f'Loading encoder network from "{encoder_ckpt_filename}"...')
    encoder_net = Module.from_checkpoint(to_absolute_path(encoder_ckpt_filename))
    encoder_net = encoder_net.eval().to(device).to(memory_format=torch.channels_last)

    if cfg.generation.inference_mode in ["sfm", "sfm_two_stage"]:
        denoiser_ckpt_filename = cfg.generation.io.denoiser_ckpt_filename
        logger0.info(f'Loading residual network from "{denoiser_ckpt_filename}"...')
        denoiser_net = Module.from_checkpoint(to_absolute_path(denoiser_ckpt_filename))
        denoiser_net = denoiser_net.eval().to(device).to(memory_format=torch.channels_last)
    else:
        denoiser_net = None

    if cfg.generation.perf.force_fp16:
        encoder_net.use_fp16 = True
        denoiser_net.use_fp16 = True

    # Reset since we are using a different mode.
    if cfg.generation.perf.use_torch_compile:
        torch._dynamo.reset()
        encoder_net = torch.compile(encoder_net, mode="reduce-overhead")
        if denoiser_net:
            denoiser_net = torch.compile(denoiser_net, mode="reduce-overhead")
    networks = {'denoiser_net': denoiser_net, 'encoder_net': encoder_net}

    # Partially instantiate the sampler based on the configs
    if cfg.generation.inference_mode in ["sfm", "sfm_two_stage"]:
        if cfg.generation.learnable_sigma:
            sampler_fn = SFM_Euler_sampler_Adaptive_Sigma
        else:
            sampler_fn = SFM_Euler_sampler
    elif cfg.generation.inference_mode == "sfm_encoder":
        sampler_fn = SFM_encoder_sampler
    else:
        raise ValueError(f"Unknown sampling method {cfg.generation.inference_mode}")

    # Main generation definition
    def generate_fn():
        img_shape_y, img_shape_x = img_shape
        with nvtx.annotate("generate_fn", color="green"):
            all_images = []
            for batch_seeds in tqdm.tqdm(rank_batches, unit="batch", disable=(dist.rank != 0)):
                batch_size = len(batch_seeds)
                if batch_size == 0:
                    continue
                rnd = StackedRandomGenerator(device, batch_seeds)
                with nvtx.annotate(f"{cfg.generation.inference_mode} model", color="rapids"):
                    with torch.inference_mode():
                        images = sampler_fn(
                            networks=networks,
                            img_lr=image_lr,
                            randn_like=rnd.randn_like,
                            cfg=cfg.generation.sampler,
                        )
                    all_images.append(images)
            image_out = torch.cat(all_images)

            # Gather tensors on rank 0
            if dist.world_size > 1:
                if dist.rank == 0:
                    gathered_tensors = [
                        torch.zeros_like(
                            image_out, dtype=image_out.dtype, device=image_out.device
                        )
                        for _ in range(dist.world_size)
                    ]
                else:
                    gathered_tensors = None

                torch.distributed.barrier()
                gather(
                    image_out,
                    gather_list=gathered_tensors if dist.rank == 0 else None,
                    dst=0,
                )

                if dist.rank == 0:
                    return torch.cat(gathered_tensors)
                else:
                    return None
            else:
                return image_out

    # generate images
    output_path = getattr(cfg.generation.io, "output_filename", "corrdiff_output.nc")
    logger0.info(f"Generating images, saving results to {output_path}...")
    batch_size = 1
    warmup_steps = min(len(times) - 1, 2)
    # Generates model predictions from the input data using the specified
    # `generate_fn`, and save the predictions to the provided NetCDF file. It iterates
    # through the dataset using a data loader, computes predictions, and saves them along
    # with associated metadata.
    if dist.rank == 0:
        f = nc.Dataset(output_path, "w")
        # add attributes
        f.cfg = str(cfg)

    with torch.cuda.profiler.profile():
        with torch.autograd.profiler.emit_nvtx():

            data_loader = torch.utils.data.DataLoader(
                dataset=dataset, sampler=sampler, batch_size=1, pin_memory=True
            )
            time_index = -1
            if dist.rank == 0:
                writer = NetCDFWriter(
                    f,
                    lat=dataset.latitude(),
                    lon=dataset.longitude(),
                    input_channels=dataset.input_channels(),
                    output_channels=dataset.output_channels(),
                )

                # Initialize threadpool for writers
                writer_executor = ThreadPoolExecutor(
                    max_workers=cfg.generation.perf.num_writer_workers
                )
                writer_threads = []

            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)

            times = dataset.time()
            for image_tar, image_lr, index in iter(data_loader):
                time_index += 1
                if dist.rank == 0:
                    logger0.info(f"starting index: {time_index}")

                if time_index == warmup_steps:
                    start.record()

                # continue
                image_lr = (
                    image_lr.to(device=device)
                    .to(torch.float32)
                    .to(memory_format=torch.channels_last)
                )
                # expand to batch size
                image_lr = (
                    image_lr.expand(cfg.generation.seed_batch_size, -1, -1, -1).to(memory_format=torch.channels_last)
                )
                image_tar = image_tar.to(device=device).to(torch.float32)
                image_out = generate_fn()

                if dist.rank == 0:
                    batch_size = image_out.shape[0]
                    # write out data in a seperate thread so we don't hold up inferencing
                    writer_threads.append(
                        writer_executor.submit(
                            save_images,
                            writer,
                            dataset,
                            list(times),
                            image_out.cpu(),
                            image_tar.cpu(),
                            image_lr.cpu(),
                            time_index,
                            index[0],
                        )
                    )
            end.record()
            end.synchronize()
            elapsed_time = start.elapsed_time(end) / 1000.0  # Convert ms to s
            timed_steps = time_index + 1 - warmup_steps
            if dist.rank == 0:
                average_time_per_batch_element = elapsed_time / timed_steps / batch_size
                logger.info(
                    f"Total time to run {timed_steps} steps and {batch_size} members = {elapsed_time} s"
                )
                logger.info(
                    f"Average time per batch element = {average_time_per_batch_element} s"
                )

            # make sure all the workers are done writing
            if dist.rank == 0:
                for thread in list(writer_threads):
                    thread.result()
                    writer_threads.remove(thread)
                writer_executor.shutdown()

    if dist.rank == 0:
        f.close()
    logger0.info("Generation Completed.")


if __name__ == "__main__":
    main()
