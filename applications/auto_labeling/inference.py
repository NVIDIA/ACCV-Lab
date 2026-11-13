#!/usr/bin/env python

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

"""Stream calibrated video scenes through StreamPETR and export LiDAR detections."""

import argparse
import json
from pathlib import Path
import sys
import tempfile

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
DEFAULT_CONFIG = Path(__file__).resolve().parent / 'configs' / 'streampetr_r50.py'


def positive_int(value):
    value = int(value)
    if value < 1:
        raise argparse.ArgumentTypeError('must be positive')
    return value


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default=str(DEFAULT_CONFIG))
    parser.add_argument('--checkpoint', required=True, help='Trusted StreamPETR 428q checkpoint')
    parser.add_argument('--pkl-path', required=True, help='Trusted nuScenes video info pickle')
    parser.add_argument('--data-root', required=True)
    parser.add_argument('--video-base-path', help='Root for relative video paths (default: data root)')
    parser.add_argument('--output-dir', default=str(Path(__file__).resolve().parent / 'outputs'))
    parser.add_argument('--batch-size', type=positive_int, default=1, help='Parallel scenes on one GPU')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--num-samples', type=positive_int, help='Maximum number of output samples')
    parser.add_argument('--score-threshold', type=float, default=0.3)
    args = parser.parse_args(argv)
    if not 0 <= args.score_threshold <= 1:
        parser.error('--score-threshold must be between 0 and 1')
    return args


def run_multi_video_inference(args, provider, engine):
    """Yield CPU results per batch; finished/padding slots never produce labels."""
    sampler = provider.create_multi_video_sampler(args.batch_size)
    limit = min(args.num_samples or len(provider), len(provider))
    produced = 0
    for round_idx in range(sampler.total_rounds):
        sampler.set_round(round_idx)
        engine.reset_multi_video_state()
        for frame_idx in range(sampler.get_max_frames_in_round()):
            batch, first_frame = provider.get_multi_video_batch()
            valid = [
                slot
                for slot, active in enumerate(sampler.active_mask)
                if active and frame_idx < sampler.get_video_length(slot)
            ][: limit - produced]
            if produced + len(valid) < limit:
                provider.prefetch_next_batch()
            parsed = engine.parse_results(engine.infer_multi_video(batch, first_frame))
            if len(parsed) != args.batch_size:
                raise RuntimeError('Model returned an incomplete batch')
            yield [parsed[slot] for slot in valid], [batch['img_metas'][slot] for slot in valid]
            produced += len(valid)
            if produced >= limit:
                return
            provider.advance_multi_video_frame()


def save_results_json(batches, output_path, class_names, score_threshold=0.3):
    """Write incrementally and publish a complete JSON file only after success.

    Boxes use the LiDAR frame: translation is bottom center, size is (dx, dy,
    dz), yaw is in radians around z, and velocity is (vx, vy).
    """
    output_path = Path(output_path)
    temporary = None
    count = 0
    try:
        with tempfile.NamedTemporaryFile(
            mode='w', dir=output_path.parent, prefix=output_path.name + '.', suffix='.tmp', delete=False
        ) as handle:
            temporary = Path(handle.name)
            handle.write('[\n')
            for results, metadata in batches:
                if len(results) != len(metadata):
                    raise ValueError('Results and metadata must have the same length')
                for result, meta in zip(results, metadata):
                    if not (len(result['boxes_3d']) == len(result['scores_3d']) == len(result['labels_3d'])):
                        raise ValueError('Boxes, scores and labels must have the same length')
                    detections = []
                    for box, score, label in zip(
                        result['boxes_3d'], result['scores_3d'], result['labels_3d']
                    ):
                        if score < score_threshold:
                            continue
                        label = int(label)
                        if not 0 <= label < len(class_names):
                            raise ValueError(f'Unknown model class: {label}')
                        detections.append(
                            dict(
                                translation=box[:3].tolist(),
                                size=box[3:6].tolist(),
                                rotation_yaw=float(box[6]),
                                velocity=box[7:9].tolist() if len(box) >= 9 else [0.0, 0.0],
                                score=float(score),
                                label=label,
                                label_name=class_names[label],
                            )
                        )
                    sample = dict(
                        sample_idx=meta['sample_idx'],
                        scene_token=meta['scene_token'],
                        filename=meta['filename'],
                        num_detections=len(detections),
                        detections=detections,
                    )
                    if count:
                        handle.write(',\n')
                    json.dump(sample, handle, allow_nan=False)
                    count += 1
            handle.write('\n]\n')
        temporary.replace(output_path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return count


def main(argv=None):
    args = parse_args(argv)
    import torch
    from mmcv import Config
    from applications.auto_labeling.data_provider import VideoDataProvider
    from applications.auto_labeling.inference_engine import InferenceEngine

    device = torch.device(args.device)
    if device.type != 'cuda' or not torch.cuda.is_available():
        raise ValueError('Auto labeling requires a CUDA device')
    torch.cuda.set_device(device)
    for path in [args.config, args.checkpoint, args.pkl_path]:
        if not Path(path).is_file():
            raise FileNotFoundError(path)
    cfg = Config.fromfile(args.config)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    provider = VideoDataProvider(
        args.pkl_path,
        args.data_root,
        dict(cfg.img_norm_cfg),
        dict(cfg.ida_aug_conf),
        args.batch_size,
        str(device),
        args.video_base_path,
    )
    try:
        engine = InferenceEngine(args.config, args.checkpoint, str(device), args.batch_size)
        count = save_results_json(
            run_multi_video_inference(args, provider, engine),
            output_dir / 'detection_results.json',
            cfg.class_names,
            args.score_threshold,
        )
        print(f'Saved {count} samples to {output_dir / "detection_results.json"}')
    finally:
        provider.close()


if __name__ == '__main__':
    main()
