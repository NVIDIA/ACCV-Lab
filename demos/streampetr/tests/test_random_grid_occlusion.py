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

import importlib.util
from pathlib import Path

import pytest
import torch


MODULE_PATH = (
    Path(__file__).parents[1]
    / "projects"
    / "mmdet3d_plugin"
    / "models"
    / "utils"
    / "random_grid_occlusion.py"
)
SPEC = importlib.util.spec_from_file_location("random_grid_occlusion", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)
RandomGridOcclusion = MODULE.RandomGridOcclusion

SUPPORTED_ARGUMENTS = {
    "use_h": True,
    "use_w": True,
    "rotate": 1,
    "offset": False,
    "ratio": 0.5,
    "mode": 1,
}


def make_augmentation(**overrides):
    arguments = {**SUPPORTED_ARGUMENTS, "prob": 1.0, **overrides}
    return RandomGridOcclusion(**arguments)


def test_eval_mode_is_identity() -> None:
    augmentation = make_augmentation().eval()
    images = torch.randn(2, 3, 12, 16)

    assert augmentation(images) is images


def test_zero_probability_is_identity() -> None:
    augmentation = make_augmentation(prob=0.0).train()
    images = torch.randn(2, 3, 12, 16)

    assert augmentation(images) is images


def test_pattern_preserves_tensor_properties() -> None:
    torch.manual_seed(7)
    augmentation = make_augmentation().train()
    images = torch.ones(2, 3, 12, 16, dtype=torch.float64, requires_grad=True)

    result = augmentation(images)

    assert result.shape == images.shape
    assert result.dtype == images.dtype
    assert result.device == images.device
    assert torch.any(result == 0)
    assert torch.any(result == 1)
    result.sum().backward()
    assert images.grad is not None


def test_fixed_seed_is_deterministic() -> None:
    augmentation = make_augmentation().train()
    images = torch.ones(1, 1, 20, 20)

    torch.manual_seed(123)
    first = augmentation(images)
    torch.manual_seed(123)
    second = augmentation(images)

    torch.testing.assert_close(first, second)


def test_probability_controls_application_frequency() -> None:
    torch.manual_seed(2026)
    augmentation = make_augmentation(prob=0.7).train()
    images = torch.ones(1, 1, 31, 37)
    trials = 2_000

    applications = sum(not torch.equal(augmentation(images), images) for _ in range(trials))

    # Wide enough to avoid a flaky test while detecting a broken probability gate.
    assert 0.65 * trials < applications < 0.75 * trials


def test_random_sampling_produces_multiple_patterns() -> None:
    torch.manual_seed(17)
    augmentation = make_augmentation().train()
    images = torch.ones(1, 1, 23, 29)

    patterns = {augmentation(images).cpu().numpy().tobytes() for _ in range(100)}

    assert len(patterns) > 20


def test_probability_schedule_retains_original_api() -> None:
    augmentation = make_augmentation(prob=0.8)

    augmentation.set_prob(epoch=5, max_epoch=10)

    assert augmentation.probability == pytest.approx(0.4)


@pytest.mark.parametrize(
    ("argument", "value", "expected_message"),
    [
        ("use_h", False, "use_h and use_w must both be True"),
        ("use_w", False, "use_h and use_w must both be True"),
        ("rotate", 0, "rotate must be 1"),
        ("rotate", 2, "rotate must be 1"),
        ("offset", True, "offset must be False"),
        ("ratio", 0.25, "ratio must be 0.5"),
        ("mode", 0, "mode must be 1"),
    ],
)
def test_unsupported_configuration_fails_cleanly(
    argument: str, value, expected_message: str
) -> None:
    arguments = {**SUPPORTED_ARGUMENTS, argument: value}

    with pytest.raises(NotImplementedError, match=expected_message):
        RandomGridOcclusion(**arguments)


@pytest.mark.parametrize("probability", [-0.1, 1.1])
def test_invalid_probability_is_rejected(probability: float) -> None:
    with pytest.raises(ValueError, match="prob must be between 0 and 1"):
        make_augmentation(prob=probability)


def test_non_nchw_input_is_rejected() -> None:
    augmentation = make_augmentation()

    with pytest.raises(ValueError, match="NCHW"):
        augmentation(torch.ones(3, 12, 12))
