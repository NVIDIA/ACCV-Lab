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

from accvlab_build_config.helpers.yaml_config import YamlConfigError, load_yaml_config


class YamlConfigTest(unittest.TestCase):
    def test_os_error_is_wrapped(self):
        config_path = Path("/ACCV-Lab/packages/lane_helpers/cpp_unit_tests.yaml")
        with mock.patch.object(Path, "is_file", return_value=True):
            with mock.patch.object(Path, "open", side_effect=OSError("Permission denied")):
                with self.assertRaises(YamlConfigError) as raised:
                    load_yaml_config(config_path)

        self.assertIn("Unable to read YAML configuration file", str(raised.exception))
        self.assertIsInstance(raised.exception.__cause__, OSError)

    def test_unicode_decode_error_is_wrapped(self):
        config_path = Path("/ACCV-Lab/packages/lane_helpers/cpp_unit_tests.yaml")
        decode_error = UnicodeDecodeError("utf-8", b"\xff", 0, 1, "invalid start byte")
        with mock.patch.object(Path, "is_file", return_value=True):
            with mock.patch.object(Path, "open", side_effect=decode_error):
                with self.assertRaises(YamlConfigError) as raised:
                    load_yaml_config(config_path)

        self.assertIn("Unable to read YAML configuration file", str(raised.exception))
        self.assertIs(raised.exception.__cause__, decode_error)


if __name__ == "__main__":
    unittest.main()
