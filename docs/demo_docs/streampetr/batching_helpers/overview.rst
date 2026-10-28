Overview
========

.. note::
  For the selection of other available demos & where to find the implementation, please refer to the 
  :doc:`demos <../../demos>` section.

The optimization of the StreamPETR model follows a similar approach as we used in the example for the
``batching-helpers`` package. Please refer to the corresponding
:doc:`example <../../../contained_package_docs_mirror/batching_helpers/docs/example>` 
for an overview of the general approach as well as an explanation of the optimization.

However, the StreamPETR implementation does not follow exactly the same design as the example, as the
task is different in that:

  - Multiple batch dimensions are present (iterating over the samples, and e.g. the decoder layers, or camera
    images).
  - The loss functions are designed differently, and the implementation has to be adapted to their interfaces.


The resulting main differences are:

  - **Matching**: Multiple batch dimensions are used for the input data (e.g. for the samples and the decoder 
    layers). However, as far as the matching is concerned, we do not care about the so defined shapes for the 
    batch dimensions, and process the data as if it had a single batch dimension (using helper functions for 
    convenient re-shaping), so that the matching implementation mostly follows the design outlined in the 
    example, but with some minor differences.
  - **Loss computation**
    
    - Multiple batch dimensions are present in the input data. Here 
      
      - For the StreamPETR head, the dimensions are relevant as we need to reduce over one of them (the 
        samples), while retaining the other dimension (the decoder layers). Therefore, the dimensions 
        are retained until the end of the loss computation, where the reduction is performed.
      - For the Focal head, the dimensions are not relevant (they represent samples & cameras inside the 
        samples), and we can flatten them away early in the implementation.
    - The loss functions are designed to expect no batch dimension and instead, all the objects are processed
      as one list containing all the objects. To use the batching helpers, we need a wrapper around the loss
      functions that will handle the conversion between the formats, and apply the reduction only over the 
      dimensions where it is needed.

Similar to the example, we will focus on the matching implementation first, and then dive into the loss 
computation implementation.

See the :doc:`evaluation` for the measured performance gains and instructions for comparing the
baseline and batched implementations.
