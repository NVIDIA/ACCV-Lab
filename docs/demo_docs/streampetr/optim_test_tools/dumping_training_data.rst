Dumping the Training Data (for Debugging/Demonstration)
=======================================================

For debugging/demonstration purposes, the training data can be dumped to a file. This is not strictly needed
to use the DALI pipeline, but it is useful to have the option to do so. For that, we use the
:class:`accvlab.optim_test_tools.TensorDumper`. Please also see the 
:doc:`../../../contained_package_docs_mirror/optim_test_tools/docs/index` documentation for more details.

Here, we will focus on two points regarding the dumping:
  - How to enable the dumping and configure the dumping (e.g. dump custom data types).
  - How to perform the dumping, including some advanced features (e.g. draw bounding boxes into the images
    using custom code which is only executed when the dumping is enabled).

Enable & Configure the Dumping
------------------------------

Note that one key point of the ``optim_test_tools`` package is that the functionality can be centrally
controlled (enabled/disabled and configured; also see the
:doc:`../../../contained_package_docs_mirror/optim_test_tools/docs/index` documentation). Here, this is done 
in the main training script (``demos/streampetr/tools/train.py``) as demonstrated in the following snippet.

.. note-literalinclude:: ../../../../demos/streampetr/tools/train.py
   :language: python
   :caption: demos/streampetr/tools/train.py
   :linenos:
   :lineno-match:
   :start-at: # ===== Profiling and Dumping Configuration =====
   :end-at: # ===== End of Profiling and Dumping Configuration =====

Note that the configuration whether to perform dumping and for how many iterations could be easily done using 
config files as well. However, as this is a debugging tool, we keep the configuration in the main training 
script, and leave it disabled by default.

Perform the Dumping
-------------------

The dumping itself is performed in the ``forward_train`` method of the model 
(``demos/streampetr/projects/mmdet3d_plugin/models/detectors/petr3d.py``). This demonstrates how we can 
separate the actual dumping from the configuration. Note that here, we can focus on the dumping itself, 
without caring where or how the configuration is done.

.. note-literalinclude:: ../../../../demos/streampetr/projects/mmdet3d_plugin/models/detectors/petr3d.py
   :language: python
   :caption: demos/streampetr/projects/mmdet3d_plugin/models/detectors/petr3d.py
   :linenos:
   :lineno-match:
   :start-at: # ===== Data dumping =====
   :end-at: # ===== End of data dumping =====

The function ``add_drawn_bboxes_to_dump`` is a custom function that draws the bounding boxes into the images
and adds them to the dump. Note that it is only called if the dumper is enabled, avoiding unnecessary 
overhead. The function is implemented as follows:

.. note-literalinclude:: ../../../../demos/streampetr/projects/mmdet3d_plugin/models/detectors/petr3d.py
   :language: python
   :caption: demos/streampetr/projects/mmdet3d_plugin/models/detectors/petr3d.py
   :linenos:
   :lineno-match:
   :start-at: # ===== Custom code for dumping the bounding boxes =====
   :end-at: # ===== End of custom code for dumping the bounding boxes =====

As demonstrated in the snippet above, the function is called in the ``forward_train`` method of the model.
Here, it is only called if the dumper is enabled, avoiding unnecessary overhead.