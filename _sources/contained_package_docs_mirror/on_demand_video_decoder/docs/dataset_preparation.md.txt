# Dataset Preparation

This section describes how to set up the NuScenes Mini dataset for testing and profiling the video decoder,
as well as for potential training purposes.

## Dataset Overview

We use **NuScenes Mini** as the test dataset. NuScenes is a large-scale autonomous driving dataset that 
provides multi-modal sensor data including camera images, LiDAR, and radar. The mini version contains a subset 
of the full dataset, making it ideal for testing and development.

- **Dataset Source**: [NuScenes Official Website](https://www.nuscenes.org/nuscenes)
- **Mini Dataset**: [Kaggle NuScenes Mini](https://www.kaggle.com/datasets/aadimator/nuscenes-mini/data)

## Download Dataset

Download the dataset, e.g. using the Kaggle tool:

```python
import kagglehub

# Download latest version
path = kagglehub.dataset_download("aadimator/nuscenes-mini")

print("Path to dataset files:", path)
# Default download path: ~/.cache/kagglehub/datasets/aadimator/nuscenes-mini/
```

## Convert Images to Video

The NuScenes dataset contains individual JPEG images that need to be converted to a video format for
use in the video decoder.

> **ℹ️ Note**: The converter scripts are designed to be run in an environment where a fully-functional 
> `FFmpeg` binary is available. This is not the case in our default docker image, where only a minimal 
> version of `FFmpeg` is set up. Please use an environment with a fully-functional `FFmpeg` binary to run 
> the converter scripts.

> **ℹ️ Note**: The examples below assume that the working directory is the root directory of the accvlab 
> package. For other working directories, you need to adjust the paths accordingly. The scripts are located at
> `packages/on_demand_video_decoder/scripts/` in the accvlab package.

We provide scripts that will combine and convert the images from both the samples & sweeps to the video 
format, and modify the sample data metadata to point to the correct video filename & frame ID corresponding to 
the image originally used in the sample.

This conversion can be performed in two steps:

### Step 1: Convert sample & sweep images to video format

```bash
python packages/on_demand_video_decoder/scripts/generate_nuscenes_video_with_sweeps.py \
    --nuscenes_root /path/to/your/nuscenes/dataset/ \
    --fps 12 \
    --gop_size 30 \
    --interpolation_num_frames 0 \
    --video_sub_dir path_to_subdirectory_for_generated_videos
```
Note that:
- The parameter `--fps` sets the FPS information in the metadata for the generated videos (and has no other 
  effect).
- The parameter `--gop_size` sets the GOP size for the generated videos.
- The parameter `--interpolation_num_frames` sets the number of additional frames to add between existing 
  frames. A simple linear interpolation is used in this case. The default value is 0, which means no 
  interpolation is performed.
- The version of the NuScenes dataset does not need to be specified. The script does not access the metadata.
  Instead, it automatically processes all available samples and sweeps, using information contained in the 
  file paths (including filenames & timestamps) as a basis for grouping the images into videos. This means
  that if the full NuScenes dataset is present, the script will process all available samples and sweeps.
  For development, it is recommended to use the script in a dataset containing only data from the mini 
  version.

You can call the script with `-h` to see all available options.

```{note}
The example scripts provided for the On-Demand Video Decoder package (see
[Sample Code Documentation](sample) and 
[PyTorch Integration Examples](pytorch_integration_examples/index)) assume that:
  - The path to the NuScenes dataset is `/data/nuscenes`
  - The path to the output directory is `/data/nuscenes/video_samples` (set `--video_sub_dir video_samples`
    when running this script to place the generated videos there)
```

Output layout for generated videos:
- Per-sequence folders, e.g.: 
  `<path_to_nuscenes_dataset>/video_samples/n008-2018-08-30-15-16-55-0400/CAM_FRONT.mp4`
- Inside each sequence folder, files are named by camera only (`CAM_FRONT.mp4`, `CAM_BACK.mp4`, etc.).

```{note}
The data is arranged into sequences based on the descriptor in the filename such as e.g. 
`n008-2018-08-30-15-16-55-0400` in `n008-2018-08-30-15-16-55-0400__CAM_FRONT.jpg`. However, in some cases, 
multiple sequences use the same descriptor. In this case, the data is split into the individual sequences 
based on large gaps in the timestamp. In this cases, the sequence folder is named with a `__partN` suffix, 
e.g. `n015-2018-11-21-19-38-26+0800__part0`, `n015-2018-11-21-19-38-26+0800__part1`, etc.
```

Apart from the generated videos, the script will also create a json file mapping the original image
paths & filenames to the generated video paths (relative to the video file output directory, 
default is `video_samples`) and frame IDs. This mapping is used in the next step to adjust the metadata for 
the camera samples.

### Step 2: Adjust camera sample metadata

This step updates the sample metadata to point to the correct frame IDs in the correct generated video for
each camera sample. 

```bash
python packages/on_demand_video_decoder/scripts/add_nuscenes_video_meta_from_json.py \
    --nuscenes_root /path/to/your/nuscenes/dataset/ \
    --nuscenes_version v1.0-mini \
    --video_sub_dir path_to_subdirectory_for_generated_videos
```
You can call the script with `-h` to see all available options.

This script will create an updated `sample_data_video.json` file in the same directory as the original 
`sample_data.json` file. To use the updated metadata, you can rename it to `sample_data.json` and replace the 
original file. This will make the video-related metadata available in the sample data for cameras. The new 
fields added to the metadata for camera sample data are:
- `video_filename`: The filename of the video containing the image used in the sample (relative to the dataset
  root directory)
- `video_frame`: The frame index of the video containing the image used in the sample
Note that the original `filename` field is not modified and still points to the original image file.