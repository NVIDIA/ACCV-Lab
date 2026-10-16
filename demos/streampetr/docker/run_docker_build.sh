#!/bin/bash

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

# Set default values if not provided
TAG=${TAG:-accvlab}
USER_ID=${USER_ID:-$(id -u)}
USER_NAME=${USER_NAME:-$(whoami)}
GROUP_ID=${GROUP_ID:-$(id -g)}
GROUP_NAME=${GROUP_NAME:-$(id -gn)}

echo "Building Docker image with:"
echo "  TAG: $TAG"
echo "  USER_ID: $USER_ID"
echo "  USER_NAME: $USER_NAME"
echo "  GROUP_ID: $GROUP_ID"
echo "  GROUP_NAME: $GROUP_NAME"

docker build --progress auto --build-arg USER_ID=$USER_ID \
                             --build-arg USER_NAME=$USER_NAME \
                             --build-arg GROUP_ID=$GROUP_ID \
                             --build-arg GROUP_NAME=$GROUP_NAME \
                             --file Dockerfile \
                             --tag $TAG ..

