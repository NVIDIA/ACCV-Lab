# ---------------------------------------------
# Copyright (c) OpenMMLab. All rights reserved.
# ---------------------------------------------
#  Modified by Zhiqi Li
# (BEVFormer: https://github.com/fundamentalvision/BEVFormer)
# ---------------------------------------------
# Modified in StreamPETR (https://github.com/exiawsh/StreamPETR).
# ---------------------------------------------
# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Changes: - Added profiling / debug-dump helpers (NVTX/Stopwatch/TensorDumper) and use of related configuration.
#          - Added option to use DALI pipeline for data loading and pre-processing.
#          - Added extensive notes/comments throughout the file (code snippets used in documentation).
#          - Added unsharded sampling for replicated mini-dataset evaluations.
#          - Applied configured multiprocessing and OpenCV thread settings before creating workers.
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

from __future__ import division

import argparse
import copy
import mmcv
import os
import time
import torch
import warnings
import time

# ===== Obtaining singleton instances =====

#@NOTE: Obtaining the the (potentially) usedsingleton instances for classes from `optim_test_tools` package.
from accvlab.optim_test_tools import Stopwatch, NVTXRangeWrapper, TensorDumper
stopwatch = Stopwatch()
nvtx_wrp = NVTXRangeWrapper()
dumper = TensorDumper()

# ===== End of Obtaining singleton instances =====


# ===== Profiling and Dumping Configuration =====

#@NOTE 
# Set `enable_dumper` to `True` to enable the dumper. The number of iterations 
# to dump is set by `dumper_iter_count`. Note that this is hard-coded here instead
# of being part of the config because the dumper is only used for debugging purposes 
# and should be kept disabled during actual training.
ENABLE_DUMPER = False
DUMPER_ITER_COUNT = 10

#@NOTE 
# If the dumper should be enabled, do so and also register custom converters for the non-standard 
# types that need to be dumped.
if ENABLE_DUMPER:
    rank = os.environ['RANK'] if 'RANK' in os.environ else (
        os.environ['LOCAL_RANK'] if 'LOCAL_RANK' in os.environ else 0
    )
    #@NOTE 
    # Enable the dumper and set the dump directory according to the rank. This will allow dumping from all
    # worker processes, without introducing conflicts.
    dumper.enable(f"/streampetr/dumps/rank{rank}")
    #@NOTE: Perform dumping for a fixed number of iterations and then exit.
    dumper.perform_after_dump_count(DUMPER_ITER_COUNT, exit)
    #@NOTE: Register custom converters for the non-standard types that need to be dumped.
    from mmdet3d.models.detectors.base import DC
    from mmdet3d.core.bbox.structures.lidar_box3d import LiDARInstance3DBoxes
    dumper.register_custom_converter(LiDARInstance3DBoxes, lambda x: {"tensor": x.tensor})
    dumper.register_custom_converter(DC, lambda x: {"data": x.data})

# ===== End of Profiling and Dumping Configuration =====

from mmcv import Config, DictAction
from mmcv.runner import get_dist_info, init_dist, wrap_fp16_model
from os import path as osp

from mmdet import __version__ as mmdet_version
from mmdet3d import __version__ as mmdet3d_version

from mmdet3d.datasets import build_dataset
from mmdet3d.models import build_model
from mmdet3d.utils import collect_env, get_root_logger, setup_multi_processes
from mmdet.apis import set_random_seed
from mmseg import __version__ as mmseg_version
from mmcv.utils import TORCH_VERSION, digit_version

# ===== Setting up the Python Path (including DALI pipeline) =====

import sys
sys.path.append("/streampetr")
#@NOTE: Make the DALI pipeline example code available.
sys.path.append("/ACCV-Lab/packages/dali_pipeline_framework/examples")

# ===== End of Setting up the Python Path (including DALI pipeline) =====


os.environ["CUDA_LAUNCH_BLOCKING"] = "0"

#@NOTE 
# The imported functionality is not used directly here. But the imports are needed so that the corrresponding 
# classes are registered and can be found if configured to be used.
import projects.mmdet3d_plugin.core.evaluation.data_pipeline_runtime_eval_runner
import projects.mmdet3d_plugin.core.adjusted_impl.optimizer_with_time
import projects.mmdet3d_plugin.core.adjusted_impl.iter_based_runner_with_time

def parse_args():
    parser = argparse.ArgumentParser(description='Train a detector')
    parser.add_argument('config', help='train config file path')
    parser.add_argument('--work-dir', help='the dir to save logs and models')
    parser.add_argument(
        '--resume-from', help='the checkpoint file to resume from')
    parser.add_argument(
        '--no-validate',
        action='store_true',
        help='whether not to evaluate the checkpoint during training')
    group_gpus = parser.add_mutually_exclusive_group()
    group_gpus.add_argument(
        '--gpus',
        type=int,
        help='number of gpus to use '
        '(only applicable to non-distributed training)')
    group_gpus.add_argument(
        '--gpu-ids',
        type=int,
        nargs='+',
        help='ids of gpus to use '
        '(only applicable to non-distributed training)')
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument(
        '--deterministic',
        action='store_true',
        help='whether to set deterministic options for CUDNN backend.')
    parser.add_argument(
        '--options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file (deprecate), '
        'change to --cfg-options instead.')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument(
        '--autoscale-lr',
        action='store_true',
        help='automatically scale lr with the number of gpus')
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    if args.options and args.cfg_options:
        raise ValueError(
            '--options and --cfg-options cannot be both specified, '
            '--options is deprecated in favor of --cfg-options')
    if args.options:
        warnings.warn('--options is deprecated in favor of --cfg-options')
        args.cfg_options = args.options

    return args

def get_and_print_rank_and_world_size_info():
        if "RANK" in os.environ:
            rank = int(os.environ["RANK"])
        elif "LOCAL_RANK" in os.environ:
            rank = int(os.environ["LOCAL_RANK"])
            print(f"'RANK' environment variable not found; using 'LOCAL_RANK' (= {rank}) as rank instead")
        else:
            print("Neither 'RANK' nor 'LOCAL_RANK' environment variable not found; Setting `rank = 0`")
            rank = 0

        if "LOCAL_RANK" in os.environ:
            local_rank = int(os.environ["LOCAL_RANK"])
        else:
            print("'LOCAL_RANK' environment variable not found; Setting `local_rank = 0`")
            local_rank = 0

        if "WORLD_SIZE" in os.environ:
            world_size = int(os.environ["WORLD_SIZE"])
        elif "LOCAL_WORLD_SIZE" in os.environ:
            world_size = int(os.environ["LOCAL_WORLD_SIZE"])
            print(f"'WORLD_SIZE' environment variable not found; using 'LOCAL_WORLD_SIZE' (= {world_size}) as world size instead")
        else:
            print("Neither 'WORLD_SIZE' nor 'LOCAL_WORLD_SIZE' environment variable not found; Setting `world_size = 1`")
            world_size = 1

        return local_rank, rank, world_size



def get_dali_pipeline_distributed_args(local_rank, rank, world_size, replicate_dataset_per_gpu=False):
    """Return the pipeline's GPU assignment and data-sharding arguments.

    The replication option was added specifically for runtime evaluations with
    NuScenes mini. Setting ``replicate_dataset_per_gpu=True`` lets each GPU sample
    the full dataset, requiring only ``batch_size`` sequence groups per GPU.
    GPU assignment is unchanged. With reproducible seeds enabled, ``rank=0`` also
    gives every GPU the same augmentation seed.
    """
    return dict(
        device_id=local_rank,
        rank=0 if replicate_dataset_per_gpu else rank,
        world_size=1 if replicate_dataset_per_gpu else world_size,
    )


def main():
    
    args = parse_args()

    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    # Apply the configured multiprocessing start method and thread limits before
    # creating the training and data-loader processes.
    setup_multi_processes(cfg)

    # import modules from string list.
    if cfg.get('custom_imports', None):
        from mmcv.utils import import_modules_from_strings
        import_modules_from_strings(**cfg['custom_imports'])

    # import modules from plguin/xx, registry will be updated
    if hasattr(cfg, 'plugin'):
        if cfg.plugin:
            import importlib
            if hasattr(cfg, 'plugin_dir'):
                plugin_dir = cfg.plugin_dir
                _module_dir = os.path.dirname(plugin_dir)
                _module_dir = _module_dir.split('/')
                _module_path = _module_dir[0]

                for m in _module_dir[1:]:
                    _module_path = _module_path + '.' + m
                print(_module_path)
                plg_lib = importlib.import_module(_module_path)
            else:
                # import dir is the dirpath for the config file
                _module_dir = os.path.dirname(args.config)
                _module_dir = _module_dir.split('/')
                _module_path = _module_dir[0]
                for m in _module_dir[1:]:
                    _module_path = _module_path + '.' + m
                print(_module_path)
                plg_lib = importlib.import_module(_module_path)

            from projects.mmdet3d_plugin.core.apis.train import custom_train_model


    # ===== Get rank and world size =====
    #@NOTE 
    # Get the rank and world size of the current process. This is needed for the following:
    #   - The DALI pipeline, which relies on this information to correctly set up sharding, and selecting
    #     the correct device ID for the pipeline.
    #   - The stopwatch, which only runs for one process (`rank==0`).
    local_rank, rank, world_size = get_and_print_rank_and_world_size_info()
    # ===== End of Get rank and world size =====


    # ===== Runtime Measurement & Profiling Configuration =====
    #@NOTE
    # Whether to use the runtime measurements and the nvtx ranges is configurable. Here, we read the
    # configuration from the config file and enable the corresponding signletons if needed.
    runtime_meas_cfg = cfg.get("runtime_meas_config", None)
    if runtime_meas_cfg is not None:
        stopwatch_cfg = runtime_meas_cfg.stopwatch
        nvtx_rng_wrp_cfg = runtime_meas_cfg.nvtx_range_wrapper
        if stopwatch_cfg.enable and rank == 0:
            stopwatch.enable(stopwatch_cfg.num_warmup_iters, stopwatch_cfg.print_every_n_iters, stopwatch_cfg.do_cuda_sync)
        if nvtx_rng_wrp_cfg.enable:
            nvtx_wrp.enable(nvtx_rng_wrp_cfg.sync_on_push, nvtx_rng_wrp_cfg.sync_on_pop, nvtx_rng_wrp_cfg.keep_track_of_range_order)
    # ===== End of Runtime Measurement & Profiling Configuration =====


    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True

    # work_dir is determined in this priority: CLI > segment in file > filename
    if args.work_dir is not None:
        # update configs according to CLI args if args.work_dir is not None
        cfg.work_dir = args.work_dir
    elif cfg.get('work_dir', None) is None:
        # use config filename as default work_dir if cfg.work_dir is None
        cfg.work_dir = osp.join('./work_dirs',
                                osp.splitext(osp.basename(args.config))[0])
    # if args.resume_from is not None:
    if args.resume_from is not None and osp.isfile(args.resume_from):
        cfg.resume_from = args.resume_from
    if args.gpu_ids is not None:
        cfg.gpu_ids = args.gpu_ids
    else:
        cfg.gpu_ids = range(1) if args.gpus is None else range(args.gpus)
    if digit_version(TORCH_VERSION) == digit_version('1.8.1') and cfg.optimizer['type'] == 'AdamW':
        cfg.optimizer['type'] = 'AdamW2' # fix bug in Adamw
    if args.autoscale_lr:
        # apply the linear scaling rule (https://arxiv.org/abs/1706.02677)
        cfg.optimizer['lr'] = cfg.optimizer['lr'] * len(cfg.gpu_ids) / 8

    # init distributed env first, since logger depends on the dist info.
    if args.launcher == 'none':
        distributed = False
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)
        # re-set gpu_ids with distributed training mode
        _, world_size = get_dist_info()
        cfg.gpu_ids = range(world_size)

    # create work_dir
    mmcv.mkdir_or_exist(osp.abspath(cfg.work_dir))
    # dump config
    cfg.dump(osp.join(cfg.work_dir, osp.basename(args.config)))
    # init the logger before other steps
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    log_file = osp.join(cfg.work_dir, f'{timestamp}.log')
    # specify logger name, if we still use 'mmdet', the output info will be
    # filtered and won't be saved in the log_file
    # TODO: ugly workaround to judge whether we are training det or seg model
    if cfg.model.type in ['EncoderDecoder3D']:
        logger_name = 'mmseg'
    else:
        logger_name = 'mmdet'
    logger = get_root_logger(
        log_file=log_file, log_level=cfg.log_level, name=logger_name)

    # init the meta dict to record some important information such as
    # environment info and seed, which will be logged
    meta = dict()
    # log env info
    env_info_dict = collect_env()
    env_info = '\n'.join([(f'{k}: {v}') for k, v in env_info_dict.items()])
    dash_line = '-' * 60 + '\n'
    logger.info('Environment info:\n' + dash_line + env_info + '\n' +
                dash_line)
    meta['env_info'] = env_info
    meta['config'] = cfg.pretty_text

    # log some basic info
    logger.info(f'Distributed training: {distributed}')
    logger.info(f'Config:\n{cfg.pretty_text}')

    # set random seeds
    if args.seed is not None:
        logger.info(f'Set random seed to {args.seed}, '
                    f'deterministic: {args.deterministic}')
        set_random_seed(args.seed, deterministic=args.deterministic)
    cfg.seed = args.seed
    meta['seed'] = args.seed
    meta['exp_name'] = osp.basename(args.config)

    model = build_model(
        cfg.model,
        train_cfg=cfg.get('train_cfg'),
        test_cfg=cfg.get('test_cfg'))

    model.init_weights()

    if cfg.get('SyncBN', False):
        import torch.nn as nn
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
        logger.info("Using SyncBN")
        
    logger.info(f'Model:\n{model}')


    # ===== Get Dataset or DALI Output Iterator =====
    #@NOTE: The if-branch is for the DALI pipeline, the else-branch is for the original PyTorch Dataset.
    if cfg.data_loading_config.use_dali_pipeline:

        #@NOTE 
        # Import the StreamPETR pipeline setup from the DALI pipeline example (and notify user if it is not 
        # found).
        try:
            from pipeline_setup import stream_petr_pipeline
        except ImportError:
            raise ImportError(
                "Could not import `stream_petr_pipeline` from `pipeline_setup`. When running with the DALI "
                "pipeline, please make sure that the `pipeline_setup` module from the "
                "dali_pipeline_framework example (packages/dali_pipeline_framework/examples) is in the python path, as "
                "this demo re-uses the StreamPETR pipeline setup showcased there."
            )

        #@NOTE
        # Use the same dataset root for DALI training and the PyTorch validation dataset. The NuScenes
        # version is specified separately.
        nuscenes_base_dir = cfg.data.train.data_root
        nuscenes_version = cfg.data_loading_config.nuscenes_version

        num_batches_in_epoch = cfg.num_iters_per_epoch

        #@NOTE 
        # Create the DALI pipeline. Note that the returned object is a 
        # `dali_pipeline_framework.DALIStructuredOutputIterator` object, which can be used as a drop-in replacement for 
        # a PyTorch DataLoader. This means that if DALI is used, one needs to skip the creation of a PyTorch 
        # DataLoader from this object, and use it directly instead. This is done in 
        # `projects/mmdet3d_plugin/core/apis/mmdet_train.py`.
        dali_iterator = stream_petr_pipeline.setup_dali_pipeline_stream_petr_train(
            nuscenes_base_dir, nuscenes_version, num_batches_in_epoch, config=cfg.dali_preproc_config, 
            batch_size=cfg.data.samples_per_gpu,
            **get_dali_pipeline_distributed_args(
                local_rank, rank, world_size,
                cfg.data_loading_config.get('replicate_dataset_per_gpu', False),
            ),
        )
        
        #@NOTE 
        # Store as the only element in the `datasets` list. In case DALI is not used, the same variable will 
        # contain the actual PyTorch Dataset (also as a single-element list), see else-branch below.
        datasets = [dali_iterator]
        
        #@NOTE 
        # The classes can be obtrained from the Dataset object if DALI is not used. Here, we manually
        # set the classes instead. 
        classes = [
            'car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'barrier', 'motorcycle', 'bicycle', 
            'pedestrian', 'traffic_cone'
        ]

    else:

        #@NOTE: This corresponds to the original implementation, where a PyTorch Dataset is created and used.

        datasets = [build_dataset(cfg.data.train)]
        if len(cfg.workflow) == 2:
            val_dataset = copy.deepcopy(cfg.data.val)
            # in case we use a dataset wrapper
            if 'dataset' in cfg.data.train:
                val_dataset.pipeline = cfg.data.train.dataset.pipeline
            else:
                val_dataset.pipeline = cfg.data.train.pipeline
            # set test_mode=False here in deep copied config
            # which do not affect AP/AR calculation later
            # refer to https://mmdetection3d.readthedocs.io/en/latest/tutorials/customize_runtime.html#customize-workflow  # noqa
            val_dataset.test_mode = False
            datasets.append(build_dataset(val_dataset))
        
        #@NOTE 
        # Extract the classes here, to obtain the `classes` variable, equivalent to the one created if using
        # DALI. This is done to avoid calling `datasets[0].CLASSES` below, which would not work DALI is used.
        classes = datasets[0].CLASSES
    # ===== End of Getting Dataset or DALI Output Iterator =====


    # ===== Checkpoint Configuration =====
    #@NOTE
    # Note that here, we use the previously obtained `classes` variable. In the original implementation, 
    # `dataset[0].CLASSES` is used instead.
    if cfg.checkpoint_config is not None:
        # save mmdet version, config file content and class names in
        # checkpoints as meta data
        cfg.checkpoint_config.meta = dict(
            mmdet_version=mmdet_version,
            mmseg_version=mmseg_version,
            mmdet3d_version=mmdet3d_version,
            config=cfg.pretty_text,
            CLASSES=classes,
            PALETTE=datasets[0].PALETTE  # for segmentors
            if hasattr(datasets[0], 'PALETTE') else None)
    # add an attribute for visualization convenience
    model.CLASSES = classes
    # ===== End of Checkpoint Configuration =====


    # ===== Start profiling (if running with nsys) =====
    #@NOTE: Indicate that profiling should be started if the NVTX range wrapper is enabled.
    if nvtx_wrp.is_enabled:
        if "RANK" in os.environ and int(os.environ["RANK"]) == 0:
            print("Starting profiling")
            torch.cuda.cudart().cudaProfilerStart()
    # ===== End of Start profiling (if running with nsys) =====


    # ===== Train the model =====
    #@NOTE
    # Call the training function. Note that datasets is passed to the model. As this now may be a DALI 
    # iterator, one needs to make sure that the training function is able to handle this (by using it directly
    # as a drop-in replacement for a PyTorch DataLoader, instead of creating a PyTorch DataLoader from it). 
    # This is done in `projects/mmdet3d_plugin/core/apis/mmdet_train.py`.
    custom_train_model(
        model,
        datasets,
        cfg,
        distributed=distributed,
        validate=(not args.no_validate),
        timestamp=timestamp,
        meta=meta)
    # ===== End of Train the model =====


    # ===== Stop profiling (if running with nsys) =====
    #@NOTE: Indicate that profiling should be stopped if the NVTX range wrapper is enabled.
    if nvtx_wrp.is_enabled:
        if "RANK" in os.environ and int(os.environ["RANK"]) == 0:
            print("Stopping profiling")
            torch.cuda.cudart().cudaProfilerStop()
    # ===== End of Stop profiling (if running with nsys) =====
    

    # ===== Stopwatch: Print eval times =====
    #@NOTE
    # Print the run-time measurements at the end of the training. Note that depending on the setup, the 
    # measurements will also be printed periodically (e.g. every 50 iterations). Here, we ensure that the
    # last printed measurement is for the complete training.
    stopwatch.print_eval_times()
    # ===== End of Stopwatch: Print eval times =====


if __name__ == '__main__':
    main()
