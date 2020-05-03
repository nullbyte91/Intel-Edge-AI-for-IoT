#!/usr/bin/env python3
"""
 Copyright (c) 2018 Intel Corporation.
 Permission is hereby granted, free of charge, to any person obtaining
 a copy of this software and associated documentation files (the
 "Software"), to deal in the Software without restriction, including
 without limitation the rights to use, copy, modify, merge, publish,
 distribute, sublicense, and/or sell copies of the Software, and to
 permit person to whom the Software is furnished to do so, subject to
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
import time
import socket
import json
import cv2
import numpy as np
import dlib

import logging as log
from imutils.video import FPS

from argparse import ArgumentParser
from inference import Network

# Browser and OpenCV Window toggle
CPU_EXTENSION = "/opt/intel/openvino/deployment_tools/inference_engine/lib/intel64/libcpu_extension_sse4.so"

def build_argparser():
    """
    Parse command line arguments.

    :return: command line arguments
    """
    parser = ArgumentParser()
    parser.add_argument("-m", "--model", required=True, type=str,
                        help="Path to an xml file with a trained model.")
    parser.add_argument("-i", "--input", required=True, type=str,
                        help="Path to image or video file")
    parser.add_argument("-l", "--cpu_extension", required=False, type=str,
                        default=None,
                        help="MKLDNN (CPU)-targeted custom layers."
                             "Absolute path to a shared library with the"
                             "kernels impl.")
    parser.add_argument("-d", "--device", type=str, default="CPU",
                        help="Specify the target device to infer on: "
                             "CPU, GPU, FPGA or MYRIAD is acceptable. Sample "
                             "will look for a suitable plugin for device "
                             "specified (CPU by default)")
    parser.add_argument("-pt", "--prob_threshold", type=float, default=0.2,
                        help="Probability threshold for detections filtering"
                        "(0.5 by default)")
    return parser


def infer_on_stream(args):
    """
    Initialize the inference network, stream video to network,
    and output stats and video.

    :param args: Command line arguments parsed by `build_argparser()`
    :return: None
    """
    # Initialize the Inference Engine
    infer_network = Network()

    # Set Probability threshold for detections
    prob_threshold = args.prob_threshold

    # Load the model through `infer_network`
    infer_network.load_model(args.model, args.device, CPU_EXTENSION, num_requests=2)

    current_inference, next_inference = 0, 1

    # Get a Input blob shape
    in_n, in_c, in_h, in_w = infer_network.get_input_shape()

    # Get a output blob name
    _ = infer_network.get_output_name()
    
    # Handle the input stream
    cap = cv2.VideoCapture(args.input)
    cap.open(args.input)
    _, frame = cap.read()

    fps = FPS().start()
    # Process frames until the video ends, or process is exited
    while cap.isOpened():
        # Read the next frame
        flag, frame = cap.read()
        if not flag:
            break
        
        fh = frame.shape[0]
        fw = frame.shape[1]
        key_pressed = cv2.waitKey(60)

        # Pre-process the frame
        image_resize = cv2.resize(frame, (in_w, in_h))
        image = image_resize.transpose((2,0,1))
        image = image.reshape(in_n, in_c, in_h, in_w)
        
        # Perform inference on the frame
        infer_network.exec_net_async(image, request_id=current_inference)

        # Get the output of inference
        if infer_network.wait(next_inference) == 0:
            detection = infer_network.get_output(next_inference)
            for box in detection[0][0]: # Output shape is 1x1x100x7
                conf = box[2]
                if conf >= prob_threshold:
                    xmin = int(box[3] * fw)
                    ymin = int(box[4] * fh)
                    xmax = int(box[5] * fw)
                    ymax = int(box[6] * fh)
                    cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 125, 255), 3)
        current_inference, next_inference = next_inference, current_inference

        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
        #Break if escape key pressed
        if key_pressed == 27:
            break
        
        fps.update()
    
    # Release the out writer, capture, and destroy any OpenCV windows
    cap.release()
    cv2.destroyAllWindows()
    fps.stop()
    print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))


def main():
    """
    Load the network and parse the output.

    :return: None
    """
    # set log level
    log.basicConfig(filename='example.log',level=log.CRITICAL)
    
    # Grab command line args
    args = build_argparser().parse_args()

    # Inference
    infer_on_stream(args)

if __name__ == '__main__':
    main()
