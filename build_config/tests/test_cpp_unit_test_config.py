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

import unittest
from pathlib import Path
from unittest import mock

from accvlab_build_config.helpers.cpp_unit_test_config import load_cpp_unit_test_package_configs
from accvlab_build_config.helpers.yaml_config import YamlConfigError


class CppUnitTestConfigTest(unittest.TestCase):
    def _load_package_config(self, config):
        project_root = Path("/ACCV-Lab")
        config_path = project_root / "packages" / "lane_helpers" / "cpp_unit_tests.yaml"
        with mock.patch(
            "accvlab_build_config.helpers.cpp_unit_test_config.load_yaml_config",
            return_value=config,
        ):
            return load_cpp_unit_test_package_configs(project_root, config_paths=[config_path])

    def test_loads_cuda_arch_strategy(self):
        package_configs = self._load_package_config(
            {
                "cuda_arch_strategy": "torch",
                "test_option": "BUILD_TESTS",
                "test_target": "run_tests",
            }
        )

        self.assertEqual(package_configs["lane_helpers"].cuda_arch_strategy, "torch")

    def test_missing_cuda_arch_strategy_is_rejected(self):
        with self.assertRaisesRegex(YamlConfigError, "cuda_arch_strategy"):
            self._load_package_config(
                {
                    "test_option": "BUILD_TESTS",
                    "test_target": "run_tests",
                }
            )

    def test_invalid_cuda_arch_strategy_is_rejected(self):
        with self.assertRaisesRegex(YamlConfigError, "must be one of: cmake, torch"):
            self._load_package_config(
                {
                    "cuda_arch_strategy": "invalid",
                    "test_option": "BUILD_TESTS",
                    "test_target": "run_tests",
                }
            )

    def test_explicit_empty_config_paths_skips_discovery(self):
        project_root = Path("/ACCV-Lab")
        with mock.patch(
            "accvlab_build_config.helpers.cpp_unit_test_config._discover_cpp_unit_test_config_paths"
        ) as discover:
            result = load_cpp_unit_test_package_configs(project_root, config_paths=[])

        self.assertEqual(result, {})
        discover.assert_not_called()

    def test_none_config_paths_uses_discovery(self):
        project_root = Path("/ACCV-Lab")
        discovered_path = project_root / "packages" / "lane_helpers" / "cpp_unit_tests.yaml"
        with mock.patch(
            "accvlab_build_config.helpers.cpp_unit_test_config._discover_cpp_unit_test_config_paths",
            return_value=[discovered_path],
        ) as discover:
            with mock.patch(
                "accvlab_build_config.helpers.cpp_unit_test_config._load_config_path",
                return_value=[],
            ):
                load_cpp_unit_test_package_configs(
                    project_root,
                    config_paths=None,
                    package_names=["lane_helpers"],
                )

        discover.assert_called_once_with(
            project_root,
            package_names=["lane_helpers"],
            package_config_name="cpp_unit_tests.yaml",
        )


if __name__ == "__main__":
    unittest.main()
