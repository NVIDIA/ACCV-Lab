Simple Image Decoding Step
==========================

.. seealso::
    The implementation of ``SimpleExampleImageDecoder`` can be found at 
    ``packages/dali_pipeline_framework/examples/simple_processing_step_example/simple_example_image_decoder.py``.

First, let's implement a single image decoding step. Image decoding is typically the first step performed 
after obtaining the input data. For now, we assume that there is only one image to process and we know exactly 
where in the input data structure the image is located. These assumptions limit the usability of the 
processing step (e.g. using single-camera vs. multi-camera input, or using other images). We will lift this 
limitation in the next step using functionality of :class:`~accvlab.dali_pipeline_framework.pipeline.SampleDataGroup`. 
But first, let's have a look at ``SimpleExampleImageDecoder``.

.. note-literalinclude:: ../../../examples/simple_processing_step_example/simple_example_image_decoder.py
   :language: python
   :linenos:
   :lineno-match:
   :caption: packages/dali_pipeline_framework/examples/simple_processing_step_example/simple_example_image_decoder.py
   :name: simple_example_image_decoder
