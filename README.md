# Intel-edge-AI-foundation-udacity
Notes and Assignments for the Intel® Edge AI Foundation Course

### OpenVINO Workflow:
![OpenVINO_Workflow](images/openvino-4.png)

#### Model Optimizer:
The Model Optimizer helps convert models in multiple different frameworks to an Intermediate Representation, which is used with the Inference Engine. If a model is not one of the pre-converted models in the Pre-Trained Models OpenVINO™ provides, it is a required step to move onto the Inference Engine.

As part of the process, it can perform various optimizations that can help shrink the model size and help make it faster, although this will not give the model higher inference accuracy. In fact, there will be some loss of accuracy as a result of potential changes like lower precision. However, these losses in accuracy are minimized.

#### Inference Engine:
The Inference Engine runs the actual inference on a model at the Edge. It only works with the Intermediate Representations(IR) that come from the Model Optimizer, or the Intel® Pre-Trained Models in OpenVINO™ that are already in IR format.
