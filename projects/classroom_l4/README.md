# OpenVino Lesson 4 project

## Prerequisites:
To run the application in this tutorial, the OpenVINO™ toolkit and its dependencies must already be installed and verified using the included demos. Installation instructions may be found at: https://software.intel.com/en-us/articles/OpenVINO-Install-Linux.

```python
# Solve system dep and download squeezenet
bash setup.sh

```
## Run

```python
# convert frozen graph to IR
pushd model/ssd_mobilenet_v2_coco_2018_03_29/

python3 /opt/intel/openvino/deployment_tools/model_optimizer/mo_tf.py --input_model frozen_inference_graph.pb --tensorflow_object_detection_api_pipeline_config pipeline.config --reverse_input_channels --tensorflow_use_custom_operations_config /opt/intel/openvino/deployment_tools/model_optimizer/extensions/front/tf/ssd_v2_support.json

popd
```

```python
pushd src
# Object detection inference
python3 app.py -m ../model/ssd_mobilenet_v2_coco_2018_03_29/frozen_inference_graph.xml -ct 0.6 -c BLUE
```
