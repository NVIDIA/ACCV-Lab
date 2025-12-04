Examples
========

Currently, this package provides two implementations for drawing the heatmaps in general which differs in the input format corresponding two **functions** to work efficiently.

draw_heatmap
------------
This implementation is designed for the **concatenated** input format. For the details about the inputs, please refer to the `packages/draw_heatmap/examples/input_data.py` file.

.. literalinclude:: ../examples/draw_heatmap.py
   :language: python
   :caption: Concatenated input example
   :name: draw_heatmap-example

draw_heatmap_batched
--------------------
This implementation is designed for the **batched** input format. For the details about the inputs, please refer to the `packages/draw_heatmap/examples/input_data.py` file.

.. note::
   It's general to draw all classes in one heatmap. However, another option in to have one heatmap for each class. Both are supported by this function distinguishing by feeding `labels` parameter of each bounding box or not.

one-for-all classes
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. literalinclude:: ../examples/draw_heatmap_batched.py
   :language: python
   :caption: Batched input, one heatmap for all classes example
   :name: draw_heatmap_batched-example

one-for-each class
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. literalinclude:: ../examples/draw_heatmap_batched_classwise.py
   :language: python
   :caption: Batched input, one heatmap for each class example
   :name: draw_heatmap_batched-example-each-class