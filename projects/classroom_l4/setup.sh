#!/bin/bash

dModel="model"

function depSolver()
{
    # System dep
    sudo apt-get update
    sudo apt-get install wget 

    # OpenVino tensorflow dep
    pushd /opt/intel/openvino_2019.3.376/deployment_tools/model_optimizer/install_prerequisites/
    bash install_prerequisites_tf.sh
    popd 
}

function downloadSqueezeNet()
{
    wget --directory-prefix='model' http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v2_coco_2018_03_29.tar.gz
    pushd model && tar -xvf *.tar.gz 
    rm -rf ssd_mobilenet_v2_coco_2018_03_29.tar.gz
    popd 
}

# Main

# Solve system dep and openVINO tensorflow dep
depSolver

# Download squeezeNet model
downloadSqueezeNet


