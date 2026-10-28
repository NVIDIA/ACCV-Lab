# Copyright (c) OpenMMLab. All rights reserved.
#
# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Based on: mmcv/runner/iter_based_runner.py (open-mmlab/mmcv, IterBasedRunner).
# Changes: Added runtime measurements (as well as detailed comments and notes).
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

import os.path as osp
import platform
import shutil
import time
import warnings
from typing import Callable, Dict, List, Optional, Tuple, Union, no_type_check

import torch
from torch.optim import Optimizer
from torch.utils.data import DataLoader

import mmcv
from mmcv.runner.base_runner import BaseRunner
from mmcv.runner.builder import RUNNERS
from mmcv.runner.checkpoint import save_checkpoint
from mmcv.runner.hooks import IterTimerHook
from mmcv.runner.utils import get_host_info

'''
This implementation (``IterBasedRunnerWithTime``) corresponds to the original implementation 
(``IterBasedRunner`` from the ``mmcv`` package), but with added runtime measurements.
'''

#@NOTE
# Here, we import the Stopwatch and get the singleton instance. Note that the stopwatch is not enabled here.
# Instead, this is done in the main training script, so that whether to use the stopwatch is determined 
# centrally in the script and can be easily changed for all parts of the code.
from accvlab.optim_test_tools import Stopwatch
stopwatch = Stopwatch()

class IterLoader:

    def __init__(self, dataloader: DataLoader):
        self._dataloader = dataloader
        self.iter_loader = iter(self._dataloader)
        self._epoch = 0

    @property
    def epoch(self) -> int:
        return self._epoch

    def __next__(self):
        try:
            data = next(self.iter_loader)
        except StopIteration:
            self._epoch += 1
            if hasattr(self._dataloader.sampler, 'set_epoch'):
                self._dataloader.sampler.set_epoch(self._epoch)
            time.sleep(2)  # Prevent possible deadlock during epoch transition
            self.iter_loader = iter(self._dataloader)
            data = next(self.iter_loader)

        return data

    def __len__(self):
        return len(self._dataloader)


@RUNNERS.register_module()
class IterBasedRunnerWithTime(BaseRunner):
    """Iteration-based Runner.

    This runner train models iteration by iteration.
    """

    # ===== Training function & timing =====

    def train(self, data_loader, **kwargs):

        #@NOTE 
        # We use the stopwatch throughout this method to time different parts of the training loop.
        # Also see `run()` method for further stopwatch usage (measuring CPU usage, 
        # indicating the end of an iteration).

        #@NOTE: Start the measurement for the training iteration.
        stopwatch.start_meas("IterBasedRunnerWithTime.train")
        self.model.train()
        self.mode = 'train'
        self.data_loader = data_loader
        self._epoch = data_loader.epoch
        data_batch = next(data_loader)
        self.data_batch = data_batch
        self.call_hook('before_train_iter')
        #@NOTE: Start the measurement for the training forward step. 
        stopwatch.start_meas("Training step")
        outputs = self.model.train_step(data_batch, self.optimizer, **kwargs)
        #@NOTE: End the measurement for the training forward step.
        stopwatch.end_meas("Training step")
        if not isinstance(outputs, dict):
            raise TypeError('model.train_step() must return a dict')
        if 'log_vars' in outputs:
            self.log_buffer.update(outputs['log_vars'], outputs['num_samples'])
        self.outputs = outputs
        self.call_hook('after_train_iter')
        del self.data_batch
        self._inner_iter += 1
        self._iter += 1
        #@NOTE: End the measurement for the training iteration.
        stopwatch.end_meas("IterBasedRunnerWithTime.train")

        #@NOTE
        # Note that this function does not explicitly call the optimizer step. Instead, the optimizer step is
        # added as a hook, and so is called implicitly here. Further runtime measurements & nvtx ranges are 
        # added to the optimizer (see `optimizer_with_time.py` in the same directory as this file).

    # ===== End of Training function & timing =====


    @torch.no_grad()
    def val(self, data_loader, **kwargs):
        self.model.eval()
        self.mode = 'val'
        self.data_loader = data_loader
        data_batch = next(data_loader)
        self.data_batch = data_batch
        self.call_hook('before_val_iter')
        outputs = self.model.val_step(data_batch, **kwargs)
        if not isinstance(outputs, dict):
            raise TypeError('model.val_step() must return a dict')
        if 'log_vars' in outputs:
            self.log_buffer.update(outputs['log_vars'], outputs['num_samples'])
        self.outputs = outputs
        self.call_hook('after_val_iter')
        del self.data_batch
        self._inner_iter += 1

    def run(self,
            data_loaders: List[DataLoader],
            workflow: List[Tuple[str, int]],
            max_iters: Optional[int] = None,
            **kwargs) -> None:
        """Start running.

        Args:
            data_loaders (list[:obj:`DataLoader`]): Dataloaders for training
                and validation.
            workflow (list[tuple]): A list of (phase, iters) to specify the
                running order and iterations. E.g, [('train', 10000),
                ('val', 1000)] means running 10000 iterations for training and
                1000 iterations for validation, iteratively.
        """
        assert isinstance(data_loaders, list)
        assert mmcv.is_list_of(workflow, tuple)
        assert len(data_loaders) == len(workflow)
        if max_iters is not None:
            warnings.warn(
                'setting max_iters in run is deprecated, '
                'please set max_iters in runner_config', DeprecationWarning)
            self._max_iters = max_iters
        assert self._max_iters is not None, (
            'max_iters must be specified during instantiation')

        work_dir = self.work_dir if self.work_dir is not None else 'NONE'
        self.logger.info('Start running, host: %s, work_dir: %s',
                         get_host_info(), work_dir)
        self.logger.info('Hooks will be executed in the following order:\n%s',
                         self.get_hook_info())
        self.logger.info('workflow: %s, max: %d iters', workflow,
                         self._max_iters)
        self.call_hook('before_run')

        iter_loaders = [IterLoader(x) for x in data_loaders]

        self.call_hook('before_epoch')

        # ===== Training loop & timing =====

        while self.iter < self._max_iters:
            #@NOTE
            # In this loop, we use the stopwatch to
            #   - Time the training loop
            #   - Measure the CPU usage of the training loop
            #   - Indicate the end of an iteration
            # Also see `train()` method in this class for further stopwatch usage.

            #@NOTE
            # A single measurement can be configured to also measure the CPU usage. We want to measure
            # the CPU usage for the training loop, and set it up here by (referenced by the name of the 
            # measurement).
            stopwatch.set_cpu_usage_meas_name("training loop")
            for i, flow in enumerate(workflow):
                #@NOTE
                # We start the measurement for the training loop (including CPU usage measurement, see above).
                #
                # Note that in the loop, the `train()` method is called internally.
                stopwatch.start_meas("training loop")
                self._inner_iter = 0
                mode, iters = flow
                if not isinstance(mode, str) or not hasattr(self, mode):
                    raise ValueError(
                        'runner has no method named "{}" to run a workflow'.
                        format(mode))
                iter_runner = getattr(self, mode)
                for _ in range(iters):
                    if mode == 'train' and self.iter >= self._max_iters:
                        break
                    iter_runner(iter_loaders[i], **kwargs)
                #@NOTE: We end the measurement for the training loop.
                stopwatch.end_meas("training loop")
                #@NOTE 
                # Indicate the end of an iteration. This is important, as this allows the Stopwatch to
                #   - Average the runtime correctly for individual iterations
                #   - Decide when to end the warm-up phase and start the measurements (after warm-up phase)
                #   - Decide when to auto-print the measurements (every n iterations)
                stopwatch.finish_iter()

        # ===== End of Training loop & timing =====

        time.sleep(1)  # wait for some hooks like loggers to finish
        self.call_hook('after_epoch')
        self.call_hook('after_run')

    @no_type_check
    def resume(self,
               checkpoint: str,
               resume_optimizer: bool = True,
               map_location: Union[str, Callable] = 'default') -> None:
        """Resume model from checkpoint.

        Args:
            checkpoint (str): Checkpoint to resume from.
            resume_optimizer (bool, optional): Whether resume the optimizer(s)
                if the checkpoint file includes optimizer(s). Default to True.
            map_location (str, optional): Same as :func:`torch.load`.
                Default to 'default'.
        """
        if map_location == 'default':
            device_id = torch.cuda.current_device()
            checkpoint = self.load_checkpoint(
                checkpoint,
                map_location=lambda storage, loc: storage.cuda(device_id))
        else:
            checkpoint = self.load_checkpoint(
                checkpoint, map_location=map_location)

        self._epoch = checkpoint['meta']['epoch']
        self._iter = checkpoint['meta']['iter']
        self._inner_iter = checkpoint['meta']['iter']
        if 'optimizer' in checkpoint and resume_optimizer:
            if isinstance(self.optimizer, Optimizer):
                self.optimizer.load_state_dict(checkpoint['optimizer'])
            elif isinstance(self.optimizer, dict):
                for k in self.optimizer.keys():
                    self.optimizer[k].load_state_dict(
                        checkpoint['optimizer'][k])
            else:
                raise TypeError(
                    'Optimizer should be dict or torch.optim.Optimizer '
                    f'but got {type(self.optimizer)}')

        self.logger.info(f'resumed from epoch: {self.epoch}, iter {self.iter}')

    def save_checkpoint(  # type: ignore
            self,
            out_dir: str,
            filename_tmpl: str = 'iter_{}.pth',
            meta: Optional[Dict] = None,
            save_optimizer: bool = True,
            create_symlink: bool = True) -> None:
        """Save checkpoint to file.

        Args:
            out_dir (str): Directory to save checkpoint files.
            filename_tmpl (str, optional): Checkpoint file template.
                Defaults to 'iter_{}.pth'.
            meta (dict, optional): Metadata to be saved in checkpoint.
                Defaults to None.
            save_optimizer (bool, optional): Whether save optimizer.
                Defaults to True.
            create_symlink (bool, optional): Whether create symlink to the
                latest checkpoint file. Defaults to True.
        """
        if meta is None:
            meta = {}
        elif not isinstance(meta, dict):
            raise TypeError(
                f'meta should be a dict or None, but got {type(meta)}')
        if self.meta is not None:
            meta.update(self.meta)
            # Note: meta.update(self.meta) should be done before
            # meta.update(epoch=self.epoch + 1, iter=self.iter) otherwise
            # there will be problems with resumed checkpoints.
            # More details in https://github.com/open-mmlab/mmcv/pull/1108
        meta.update(epoch=self.epoch + 1, iter=self.iter)

        filename = filename_tmpl.format(self.iter + 1)
        filepath = osp.join(out_dir, filename)
        optimizer = self.optimizer if save_optimizer else None
        save_checkpoint(self.model, filepath, optimizer=optimizer, meta=meta)
        # in some environments, `os.symlink` is not supported, you may need to
        # set `create_symlink` to False
        if create_symlink:
            dst_file = osp.join(out_dir, 'latest.pth')
            if platform.system() != 'Windows':
                mmcv.symlink(filename, dst_file)
            else:
                shutil.copy(filepath, dst_file)

    def register_training_hooks(self,
                                lr_config,
                                optimizer_config=None,
                                checkpoint_config=None,
                                log_config=None,
                                momentum_config=None,
                                custom_hooks_config=None):
        """Register default hooks for iter-based training.

        Checkpoint hook, optimizer stepper hook and logger hooks will be set to
        `by_epoch=False` by default.

        Default hooks include:

        +----------------------+-------------------------+
        | Hooks                | Priority                |
        +======================+=========================+
        | LrUpdaterHook        | VERY_HIGH (10)          |
        +----------------------+-------------------------+
        | MomentumUpdaterHook  | HIGH (30)               |
        +----------------------+-------------------------+
        | OptimizerStepperHook | ABOVE_NORMAL (40)       |
        +----------------------+-------------------------+
        | CheckpointSaverHook  | NORMAL (50)             |
        +----------------------+-------------------------+
        | IterTimerHook        | LOW (70)                |
        +----------------------+-------------------------+
        | LoggerHook(s)        | VERY_LOW (90)           |
        +----------------------+-------------------------+
        | CustomHook(s)        | defaults to NORMAL (50) |
        +----------------------+-------------------------+

        If custom hooks have same priority with default hooks, custom hooks
        will be triggered after default hooks.
        """
        if checkpoint_config is not None:
            checkpoint_config.setdefault('by_epoch', False)  # type: ignore
        if lr_config is not None:
            lr_config.setdefault('by_epoch', False)  # type: ignore
        if log_config is not None:
            for info in log_config['hooks']:
                info.setdefault('by_epoch', False)
        super().register_training_hooks(
            lr_config=lr_config,
            momentum_config=momentum_config,
            optimizer_config=optimizer_config,
            checkpoint_config=checkpoint_config,
            log_config=log_config,
            timer_config=IterTimerHook(),
            custom_hooks_config=custom_hooks_config)
