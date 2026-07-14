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

from pathlib import Path
from typing import Any, Dict, Union

PathLike = Union[str, Path]


class YamlConfigError(RuntimeError):
    """Raised when a YAML configuration file cannot be loaded."""


def load_yaml_config(path: PathLike) -> Dict[str, Any]:
    """Load a YAML configuration file as a dictionary.

    Empty YAML files are treated as empty dictionaries. The top-level YAML value
    must be a mapping so callers can consume configuration files predictably.
    """

    config_path = Path(path)
    if not config_path.is_file():
        raise YamlConfigError(f"YAML configuration file not found: {config_path}")

    import yaml

    try:
        with config_path.open("r", encoding="utf-8") as config_file:
            config = yaml.safe_load(config_file)
    except yaml.YAMLError as exc:
        raise YamlConfigError(f"Invalid YAML in configuration file {config_path}: {exc}") from exc
    except (OSError, UnicodeDecodeError) as exc:
        raise YamlConfigError(f"Unable to read YAML configuration file {config_path}: {exc}") from exc

    if config is None:
        return {}

    if not isinstance(config, dict):
        raise YamlConfigError(
            f"YAML configuration file {config_path} must contain a top-level mapping, "
            f"got {type(config).__name__}."
        )

    return config
