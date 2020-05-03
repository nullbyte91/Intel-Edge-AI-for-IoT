# Intel-edge-AI-foundation-udacity
Notes and Assignments for the Intel® Edge AI Foundation Course

### OpenVINO Workflow:
![OpenVINO_Workflow](images/openvino-4.png)

#### Model Optimizer:
The Model Optimizer helps convert models in multiple different frameworks to an Intermediate Representation, which is used with the Inference Engine. If a model is not one of the pre-converted models in the Pre-Trained Models OpenVINO™ provides, it is a required step to move onto the Inference Engine.

As part of the process, it can perform various optimizations that can help shrink the model size and help make it faster, although this will not give the model higher inference accuracy. In fact, there will be some loss of accuracy as a result of potential changes like lower precision. However, these losses in accuracy are minimized.

#### Inference Engine:
The Inference Engine runs the actual inference on a model at the Edge. It only works with the Intermediate Representations(IR) that come from the Model Optimizer, or the Intel® Pre-Trained Models in OpenVINO™ that are already in IR format.

### Pre-Trained Models in OpenVINO™
```python
cd /opt/intel/openvino/deployment_tools/open_model_zoo/tools/downloader
```
Within there, you'll notice a downloader.py file, and can use the -h argument with it to see available arguments. For this exercise, --name for model name, and --precisions, used when only certain precisions are desired, are the important arguments. Note that running downloader.py without these will download all available pre-trained models, which will be multiple gigabytes. You can do this on your local machine, if desired, but the workspace will not allow you to store that much information.

i.e <b>Downloading Human Pose Model</b>

```python
sudo ./downloader.py --name human-pose-estimation-0001 -o /home/workspace 
```