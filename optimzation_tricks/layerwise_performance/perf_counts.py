import numpy as np
from openvino.inference_engine import IENetwork
from openvino.inference_engine import IEPlugin

import pprint
import os
import cv2
import argparse
import sys

def main(args):
    pp = pprint.PrettyPrinter(indent=4)
    model=args.model
    device=args.device
    image_path=args.image

    # Loading model
    model_weights=model+'.bin'
    model_structure=model+'.xml'
    
    model=IENetwork(model_structure, model_weights)
    plugin = IEPlugin(device=device)

    # Loading network to device
    net = plugin.load(network=model, num_requests=1)

    # Get the name of the input node
    input_name=next(iter(model.inputs))

    # Reading and Preprocessing Image
    input_img=np.load(image_path)
    input_img=input_img.reshape(1, 28, 28)

    # Running Inference
    net.requests[0].infer({input_name:input_img})
    pp.pprint(net.requests[0].get_perf_counts())

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--model', required=True)
    parser.add_argument('--device', default='CPU')
    parser.add_argument('--image', default=None)
    
    args=parser.parse_args()
    sys.exit(main(args) or 0)
