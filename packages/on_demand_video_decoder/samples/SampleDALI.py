import numpy as np
import nvidia.dali as dali
from nvidia.dali.pipeline import Pipeline
import accvlab.on_demand_video_decoder as odv
from nvidia.dali import pipeline_def
import nvidia.dali.fn as fn
import nvidia.dali.types as types
import torch

# Params
NUM_ITERS = 100
FRAME_STEP = 5
PREFETCH_QUEUE_DEPTH = 1

# Input data
MAX_FRAME = 200
VIDEO_PATH = "./data/sample_clip"
VIDEO_FILES = [
    VIDEO_PATH + "/moving_shape_circle_h265.mp4",
    VIDEO_PATH + "/moving_shape_ellipse_h265.mp4",
    VIDEO_PATH + "/moving_shape_hexagon_h265.mp4",
    VIDEO_PATH + "/moving_shape_rect_h265.mp4",
    VIDEO_PATH + "/moving_shape_triangle_h265.mp4",
]
print(VIDEO_FILES)
NUM_VIEWS = len(VIDEO_FILES)
frame_offsets = [np.random.randint(-2, 3) for _ in range(MAX_FRAME)]


# DALI pipeline & Decoder
@pipeline_def
def video_pipeline():
    # `no_copy=False`: Copy GPU input into DALI-owned buffers. The decoder returns zero-copy RGBFrame
    # views whose backing memory may be reused by the next decode. Even after `run()` returns, DALI
    # async GPU work can still reference the fed input while Python starts the next iteration.
    frames = fn.external_source(name="frames", dtype=types.UINT8, device="gpu", no_copy=False)
    frame_id = fn.external_source(name="frame_id", dtype=types.INT32)

    frames = fn.resize(frames, resize_x=500)
    frames = fn.cast(frames, dtype=types.FLOAT)
    frames = frames * (1.0 / 255.0)

    return frames, frame_id


pipeline = video_pipeline(
    batch_size=NUM_VIEWS,
    num_threads=4,
    device_id=0,
    prefetch_queue_depth=PREFETCH_QUEUE_DEPTH,
    exec_dynamic=True,
)
pipeline.build()

decoder = odv.CreateSampleReader(num_of_set=1, num_of_file=len(VIDEO_FILES), iGpu=0)

# Running the pipeline
for i in range(NUM_ITERS):

    # ----- PREPARATIONS -----
    set_frame = lambda v: (max((i * FRAME_STEP + frame_offsets[v]), 0) % (MAX_FRAME + 1))
    frame_ids = [set_frame(v) for v in range(NUM_VIEWS)]

    # ----- DECODE DATA -----
    data_as_cai = decoder.DecodeN12ToRGB(VIDEO_FILES, frame_ids)

    # ----- FEED TO DALI PIPELINE -----
    pipeline.feed_input("frames", data_as_cai)
    pipeline.feed_input("frame_id", np.array(frame_ids, dtype=np.int32))

    # ----- PROCESS FRAMES -----
    # Run DALI on PyTorch's current CUDA stream so later torch ops are ordered after DALI work.
    torch_stream = torch.cuda.current_stream()
    frames, frame_ids_out = pipeline.run(torch_stream.cuda_stream)

    # ----- PROCESS RESULTS -----
    frames = [torch.as_tensor(f, device="cuda").clone() for f in frames]
    # print(frames)
    frame_ids_out = frame_ids_out.as_array().tolist()
    print(f"Finished iteration {i} for frame IDs: {frame_ids_out}")
    # ...
