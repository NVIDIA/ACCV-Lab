Evaluation
==========

The on-demand video decoder was used for training a StreamPETR model on the NuScenes mini dataset and 
compared to the performance to both the 
`original StreamPETR implementation (with image-based training) <https://github.com/exiawsh/StreamPETR>`_,
and in one case to OpenCV-based video training. The results are shown below.

Setup
-----

Experiment Setup
~~~~~~~~~~~~~~~~

For the video training, the demuxer-free approach is used (see 
:doc:`pytorch_integration_examples/dataloader_demuxer_free_decode` for details on this approach). Here, the 
GOP packets are extracted and stored prior to the training.

In the video training, the frames are decoded in the training process, and consequently, pre-processing is 
performed in the training process on the GPU. Note that this is not a viable optimization for the image-based 
training, as it adds significant overhead when passing the full-resolution images to the training process.


The training is performed for the NuScenes mini dataset, with the following configuration:

  - Video

    - GOP size of 30
    - No B-frames
    - Including both samples and sweeps (resulting in ~12 frames per second)
    - 1600x900 resolution (same as images)
    
  - Batch size of 16 per GPU

.. note::

  We are planning to add a demo for the On-Demand Video Decoder package in the future, including the 
  implementation of the experiments performed in this evaluation.


Hardware Setup A
~~~~~~~~~~~~~~~~

.. list-table:: System Configuration
   :header-rows: 1

   * - GPU
     - CPU
   * - 8x NVIDIA RTX 6000D
     - 2x AMD EPYC 7742 64-core Processors


Hardware Setup B
~~~~~~~~~~~~~~~~

.. list-table:: System Configuration
   :header-rows: 1

   * - GPU
     - CPU
   * - 8x NVIDIA H20
     - 2x Intel Xeon Platinum 8468V 48-core Processors


Results & Discussion
--------------------

Results
~~~~~~~

Results for both hardware systems are shown in the following tables.

.. list-table:: Runtime Comparison for Hardware Setup A
   :header-rows: 1

   * - Configuration
     - Image [ms]
     - Video: OpenCV [ms]
     - Video: Ours [ms]
     - Speedup (vs. Image)
   * - 1 GPU
     - 725
     - 1674
     - **751**
     - × 0.97
   * - 8 GPU
     - 1025
     - 2663
     - **908**
     - × 1.13


.. list-table:: Runtime Comparison for Hardware Setup B
   :header-rows: 1

   * - Configuration
     - Image [ms]
     - Video [ms]
     - Speedup
   * - 1 GPU
     - 878
     - **862**
     - × 1.02
   * - 8 GPU
     - 1310
     - **1070**
     - × 1.22


Discussion
~~~~~~~~~~

On both systems, the performance of the video-based training is comparable to the image-based training for
the 1 GPU configuration. The video training outperforms the image training for the 8 GPU configuration,
with the speedup depending on the system. However, please note that the main goal is to reduce the storage
requirements while maintaining good performance.