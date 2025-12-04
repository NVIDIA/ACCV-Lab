Flexible Image Decoding Step
============================

.. seealso::
    The implementation of :class:`~accvlab.dali_pipeline_framework.processing_steps.ImageDecoder` is the one 
    used in the core DALI pipeline framework package. It can found in the repository at 
    ``packages/dali_pipeline_framework/accvlab/dali_pipeline_framework/processing_steps/image_decoder.py``.

Next, we will have a look at the flexible image decoding step. The functionality is similar to the 
:doc:`simple_step` version, but the implementation does not make any assumptions about how many images need 
to be processed or where in the input data they are located. Instead, it makes use of 
:meth:`~accvlab.dali_pipeline_framework.pipeline.SampleDataGroup.find_all_occurrences` to automatically 
discover all images in the input data by name.

.. important::
    
    While the discovery of the images requires additional steps in the implementation, note that these
    steps are performed at DALI graph construction time (as no :class:`~nvidia.dali.pipeline.DataNode` data
    or DALI operators are involved, only "normal" Python data). Therefore, even if elaborate discovery is 
    needed, there will be no runtime cost during pipeline execution. 
    
    For example, if the input data to this flexible image decoding step is in the format as expected by the 
    :doc:`simple_step` version, the resulting DALI graph will be the same for both versions, and they
    will therefore run in the same way.


.. note-literalinclude:: ../../../accvlab/dali_pipeline_framework/processing_steps/image_decoder.py
   :language: python
   :linenos:
   :lineno-match:
   :caption: packages/dali_pipeline_framework/accvlab/dali_pipeline_framework/processing_steps/image_decoder.py
   :name: image_decoder