# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Dataset-free regression tests for StreamPETR data-loader sampling."""

import importlib
import logging
from pathlib import Path
import sys

import numpy as np
import pytest
import torch
from mmcv import Config
from mmcv.runner.iter_based_runner import IterLoader


STREAM_PETR_ROOT = Path(__file__).resolve().parents[1]
CONFIG_ROOT = STREAM_PETR_ROOT / "projects/configs/StreamPETR"


class IndexDataset:
    """Sixteen five-frame sequences; reject batch lists used as scalar indices."""

    sequence_length = 5

    def __init__(self):
        self.flag = np.repeat(np.arange(16), self.sequence_length)

    def __len__(self):
        return len(self.flag)

    def __getitem__(self, index):
        assert isinstance(index, (int, np.integer)), "Expected a scalar sample index"
        return {"index": int(index), "group": int(self.flag[index])}


@pytest.fixture(scope="module")
def loader_builder():
    sys.path.insert(0, str(STREAM_PETR_ROOT))
    try:
        return importlib.import_module("projects.mmdet3d_plugin.datasets.builder")
    finally:
        sys.path.pop(0)


@pytest.mark.parametrize("distributed,rank,world_size", [
    (False, 0, 1),
    (True, 0, 1),
    (True, 0, 2),
    (True, 1, 2),
])
@pytest.mark.parametrize("randomize", [False, True])
@pytest.mark.parametrize("workers", [0, 2])
def test_load_only_matches_training_sampling(
    loader_builder, monkeypatch, distributed, rank, world_size, randomize, workers
):
    # Simulate distributed ranks without reserving GPUs or starting a process group.
    # The sampler receives these values explicitly; its seed is fixed at zero.
    monkeypatch.setattr(loader_builder, "get_dist_info", lambda: (rank, world_size))
    dataset = IndexDataset()
    # Isolate sampler selection from differences in the configs' workload sizes.
    samples_per_gpu = 8
    loaders = []
    for name in ("Tests/dali_ref_single_gpu.py", "DALILoadOnly/ref_load_only.py"):
        cfg = Config.fromfile(str(CONFIG_ROOT / name))
        cfg.data.shuffler_sampler.randomize = randomize
        loader = loader_builder.build_dataloader(
            dataset,
            samples_per_gpu,
            workers,
            dist=distributed,
            seed=0,
            shuffler_sampler=cfg.data.shuffler_sampler,
            nonshuffler_sampler=cfg.data.nonshuffler_sampler,
            runner_type=cfg.runner,
        )
        assert isinstance(loader.batch_sampler, loader_builder.InfiniteGroupEachSampleInBatchSampler)
        assert loader.batch_sampler.randomize is randomize
        assert loader.batch_sampler.rank == rank
        assert loader.batch_sampler.num_replicas == world_size
        loaders.append(IterLoader(loader))

    previous_indices = None
    # Cross the reported dataset/loader length to catch finite-loader restarts and
    # the two-second sleep MMCV inserts at every such restart.
    for _ in range(len(dataset) + 5):
        training_batch, load_only_batch = [next(loader) for loader in loaders]
        for key in ("index", "group"):
            torch.testing.assert_close(training_batch[key], load_only_batch[key])
        indices = load_only_batch["index"]
        assert indices.shape == (samples_per_gpu,)
        assert len(torch.unique(load_only_batch["group"])) == samples_per_gpu
        if previous_indices is not None:
            continuing = previous_indices % dataset.sequence_length != dataset.sequence_length - 1
            torch.testing.assert_close(indices[continuing], previous_indices[continuing] + 1)
        previous_indices = indices
        assert all(loader.epoch == 0 for loader in loaders)


@pytest.mark.parametrize("distributed", [False, True])
def test_validation_keeps_finite_nonshuffling_sampler(loader_builder, distributed):
    cfg = Config.fromfile(str(CONFIG_ROOT / "DALILoadOnly/ref_load_only.py"))
    dataset = IndexDataset()
    loader = loader_builder.build_dataloader(
        dataset,
        cfg.data.samples_per_gpu,
        0,
        dist=distributed,
        shuffle=False,
        seed=0,
        shuffler_sampler=cfg.data.shuffler_sampler,
        nonshuffler_sampler=cfg.data.nonshuffler_sampler,
        runner_type=cfg.runner,
    )
    assert not isinstance(loader.batch_sampler, loader_builder.InfiniteGroupEachSampleInBatchSampler)
    batches = list(loader)
    torch.testing.assert_close(torch.cat([batch["index"] for batch in batches]), torch.arange(len(dataset)))


@pytest.mark.parametrize("distributed", [False, True])
def test_default_sampler_config_still_works(loader_builder, distributed):
    loader = loader_builder.build_dataloader(
        IndexDataset(), 8, 0, dist=distributed, seed=0,
        runner_type=dict(type="IterBasedRunnerWithTime"),
    )
    assert not isinstance(loader.batch_sampler, loader_builder.InfiniteGroupEachSampleInBatchSampler)
    assert next(iter(loader))["index"].shape == (8,)


@pytest.mark.parametrize("workers", [0, 2])
@pytest.mark.parametrize("randomize", [False, True])
def test_replicated_sampling_on_eight_ranks(loader_builder, monkeypatch, workers, randomize):
    cfg = Config.fromfile(str(CONFIG_ROOT / "Tests/dali_ref_8_gpu.py"))
    cfg.data.shuffler_sampler.randomize = randomize
    dataset = IndexDataset()
    reference_batches = None
    for rank in range(8):
        monkeypatch.setattr(loader_builder, "get_dist_info", lambda rank=rank: (rank, 8))
        loader = loader_builder.build_dataloader(
            dataset, cfg.data.samples_per_gpu, workers, dist=True, seed=0,
            shuffler_sampler=cfg.data.shuffler_sampler,
            nonshuffler_sampler=cfg.data.nonshuffler_sampler,
            runner_type=cfg.runner,
            replicate_dataset_per_gpu=cfg.data_loading_config.replicate_dataset_per_gpu,
        )
        assert loader.batch_sampler.rank == 0
        assert loader.batch_sampler.num_replicas == 1
        assert loader.batch_sampler.global_batch_size == 8
        assert loader.num_workers == workers
        assert loader.worker_init_fn.keywords["rank"] == rank
        iterator = iter(loader)
        batches = [next(iterator)["index"].tolist() for _ in range(12)]
        del iterator  # Release any worker processes before checking the next rank.
        if reference_batches is None:
            reference_batches = batches
        assert batches == reference_batches
        for batch in batches:
            assert len(batch) == 8
            assert len({dataset.flag[index] for index in batch}) == 8


def test_eight_rank_replication_is_opt_in(loader_builder, monkeypatch):
    cfg = Config.fromfile(str(CONFIG_ROOT / "Tests/dali_ref_8_gpu.py"))
    monkeypatch.setattr(loader_builder, "get_dist_info", lambda: (0, 8))
    # Without the demo override, the normal global sequence limit is unchanged.
    with pytest.raises(AssertionError):
        loader_builder.build_dataloader(
            IndexDataset(), 8, 0, dist=True, seed=0,
            shuffler_sampler=cfg.data.shuffler_sampler,
            runner_type=cfg.runner,
        )


def test_replication_does_not_change_validation_sharding(loader_builder, monkeypatch):
    cfg = Config.fromfile(str(CONFIG_ROOT / "Tests/dali_ref_8_gpu.py"))
    indices = []
    for rank in range(8):
        monkeypatch.setattr(loader_builder, "get_dist_info", lambda rank=rank: (rank, 8))
        loader = loader_builder.build_dataloader(
            IndexDataset(), 8, 0, dist=True, shuffle=False, seed=0,
            shuffler_sampler=cfg.data.shuffler_sampler,
            nonshuffler_sampler=cfg.data.nonshuffler_sampler,
            runner_type=cfg.runner, replicate_dataset_per_gpu=True,
        )
        assert loader.sampler.rank == rank
        assert loader.sampler.num_replicas == 8
        indices.extend(index for batch in loader for index in batch["index"].tolist())
    assert indices == list(range(len(IndexDataset())))


@pytest.mark.parametrize("replication", [None, False, True])
def test_training_api_forwards_replication_option(loader_builder, monkeypatch, replication):
    training_api = importlib.import_module("projects.mmdet3d_plugin.core.apis.mmdet_train")
    cfg = Config.fromfile(str(CONFIG_ROOT / "Tests/dali_ref_8_gpu.py"))
    cfg.seed = 0
    cfg.gpu_ids = list(range(8))
    if replication is None:
        cfg.data_loading_config.pop("replicate_dataset_per_gpu")
    else:
        cfg.data_loading_config.replicate_dataset_per_gpu = replication
    captured = {}

    class LoaderCaptured(Exception):
        pass

    def capture_loader(*args, **kwargs):
        captured.update(kwargs)
        raise LoaderCaptured

    monkeypatch.setattr(training_api, "build_dataloader", capture_loader)
    # Avoid creating the training logger's output file in the checkout.
    monkeypatch.setattr(training_api, "get_root_logger", lambda *args, **kwargs: logging.getLogger(__name__))
    # Stop at loader construction: no model initialization, DDP or training.
    with pytest.raises(LoaderCaptured):
        training_api.custom_train_detector(object(), IndexDataset(), cfg, distributed=True)
    assert captured["replicate_dataset_per_gpu"] is bool(replication)
    assert captured["dist"] is True


@pytest.fixture(scope="module")
def training_script(loader_builder):
    sys.path.insert(0, str(STREAM_PETR_ROOT))
    try:
        return importlib.import_module("tools.train")
    finally:
        sys.path.pop(0)


@pytest.mark.parametrize("world_size", [1, 2, 8])
def test_dali_replication_preserves_devices(training_script, world_size):
    from accvlab.dali_pipeline_framework.inputs import SequenceSampler

    reference_batches = None
    for rank in range(world_size):
        # Also distinguish a node-local device ID from a global distributed rank.
        local_rank = rank % 4
        args = training_script.get_dali_pipeline_distributed_args(
            local_rank, rank, world_size, replicate_dataset_per_gpu=True
        )
        assert args == dict(device_id=local_rank, rank=0, world_size=1)
        sampler = SequenceSampler(8 * args["world_size"], [5] * 16, seed=21)
        batches = [sampler.get_next_batch_indices() for _ in range(12)]
        if reference_batches is None:
            reference_batches = batches
        assert batches == reference_batches
        assert all(len(batch) == 8 for batch in batches)


def test_dali_sharding_is_unchanged_by_default(training_script):
    for rank in range(8):
        assert training_script.get_dali_pipeline_distributed_args(rank % 4, rank, 8) == dict(
            device_id=rank % 4, rank=rank, world_size=8
        )


DALI_CONFIGS = sorted(
    list((CONFIG_ROOT / "Tests").glob("dali_*.py"))
    + list((CONFIG_ROOT / "DALILoadOnly").glob("*.py"))
    + list((CONFIG_ROOT / "Profiling").glob("*1024_bs8*8_gpu.py"))
)


@pytest.mark.parametrize("config_path", DALI_CONFIGS, ids=lambda path: path.stem)
def test_mini_dali_configs_enable_replication(config_path):
    cfg = Config.fromfile(str(config_path))
    assert cfg.data_loading_config.replicate_dataset_per_gpu is True
