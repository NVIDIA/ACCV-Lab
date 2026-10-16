#!/usr/bin/env bash

# Modified from mmdetection3d (https://github.com/open-mmlab/mmdetection3d)
# Copyright (c) OpenMMLab. All rights reserved.
# ------------------------------------------------------------------------
# Modified in StreamPETR (https://github.com/exiawsh/StreamPETR).
# ------------------------------------------------------------------------
# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Changes: Added profiling argument parsing (--profile, --profile-output) and
#          conditional nsys profiling command.
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

# Parse command line arguments
PROFILE=0
PROFILE_OUTPUT="profile_output"
POSITIONAL_ARGS=()

while [[ $# -gt 0 ]]; do
  case $1 in
    --profile)
      PROFILE=1
      shift
      ;;
    --profile-output)
      PROFILE_OUTPUT="$2"
      shift 2
      ;;
    *)
      POSITIONAL_ARGS+=("$1")
      shift
      ;;
  esac
done

set -- "${POSITIONAL_ARGS[@]}" # restore positional parameters

CONFIG=$1
GPUS=$2
NNODES=${NNODES:-1}
NODE_RANK=${NODE_RANK:-0}
PORT=${PORT:-29500}
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \

# Add nsys profile for rank 0 GPU only if profiling is enabled
if [ "$PROFILE" = "1" ] && [ "${LOCAL_RANK:-0}" = "0" ]; then
    PROFILE_CMD="nsys profile --trace=cuda,nvtx,osrt,cudnn --sample=none --cpuctxsw=none --output=$PROFILE_OUTPUT --capture-range=cudaProfilerApi"
else
    PROFILE_CMD=""
fi

$PROFILE_CMD python -m torch.distributed.launch \
    --nnodes=$NNODES \
    --node_rank=$NODE_RANK \
    --master_addr=$MASTER_ADDR \
    --use_env \
    --nproc_per_node=$GPUS \
    --master_port=$PORT \
    $(dirname "$0")/train.py --no-validate \
    $CONFIG \
    --seed 0 \
    --launcher pytorch ${@:3}
