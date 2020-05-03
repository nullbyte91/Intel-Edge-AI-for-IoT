#!/usr/bin/env python3
"""
 Copyright (c) 2018 Intel Corporation.

 Permission is hereby granted, free of charge, to any person obtaining
 a copy of this software and associated documentation files (the
 "Software"), to deal in the Software without restriction, including
 without limitation the rights to use, copy, modify, merge, publish,
 distribute, sublicense, and/or sell copies of the Software, and to
 permit persons to whom the Software is furnished to do so, subject to
 the following conditions:

 The above copyright notice and this permission notice shall be
 included in all copies or substantial portions of the Software.

 THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
 NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
 LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
 OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
 WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""

import os
import sys
import logging as log
from openvino.inference_engine import IENetwork, IECore


class Network:
    """
    Load and configure inference plugins for the specified target devices 
    and performs synchronous and asynchronous modes for the specified infer requests.
    """

    def __init__(self):
        self.plugin = None
        self.network = None
        self.input_blob = None
        self.output_blob = None 
        self.exec_network = None
        self.infer_request = None

    def load_model(self, model, device="CPU", cpu_extension=None):
        '''
        Load the model given IR files.
        Defaults to CPU as device for use in the workspace.
        Synchronous requests made within.
        '''
        model_xml = model
        model_bin = os.path.splitext(model_xml)[0] + ".bin"

        
        # Initialize the plugin
        self.plugin = IECore()

        # Read the IR as a IENetwork
        self.network = IENetwork(model=model_xml, weights=model_bin)

        # Load the IENetwork into the plugin
        self.exec_network = self.plugin.load_network(self.network, device)

        # Get the input layer
        self.input_blob = next(iter(self.network.inputs))
        self.output_blob = next(iter(self.network.outputs))
        return

    def get_input_shape(self):
        '''
        Gets the input shape of the network
        '''
        return self.network.inputs[self.input_blob].shape
    
    def get_output_name(self):
        '''
        Gets the input shape of the network
        '''
        output_name, _ = "", self.network.outputs[next(iter(self.network.outputs.keys()))]
        for output_key in self.network.outputs:
            if self.network.layers[output_key].type == "DetectionOutput":
                output_name, _ = output_key, self.network.outputs[output_key]
        
        if output_name == "":
            log.error("Can't find a DetectionOutput layer in the topology")
            exit(-1)
        return output_name

    def exec_net_async(self, image):
        '''
        Makes an asynchronous inference request, given an input image.
        '''
        self.exec_network.start_async(request_id=0, 
            inputs={self.input_blob: image})
        return

    def exec_net_sync(self, image):
        '''
        Makes an asynchronous inference request, given an input image.
        '''
        return self.exec_network.infer(inputs={self.input_blob: image})
    
    def wait(self):
        '''
        Checks the status of the inference request.
        '''
        status = self.exec_network.requests[0].wait(-1)
        return status


    def get_output(self):
        '''
        Returns a list of the results for the output layer of the network.
        '''
        return self.exec_network.requests[0].outputs[self.output_blob]
