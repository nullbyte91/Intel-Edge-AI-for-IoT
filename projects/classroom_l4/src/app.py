#!/usr/bin/env python
"""
 Copyright (C) 2018-2019 Intel Corporation
 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at
      http://www.apache.org/licenses/LICENSE-2.0
 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
"""

import argparse
import cv2
from inference import Network
import time

font = cv2.FONT_HERSHEY_SIMPLEX
INPUT_STREAM = "../input/test_video_1.mp4"
CPU_EXTENSION = "/opt/intel/openvino/deployment_tools/inference_engine/lib/intel64/libcpu_extension_sse4.so"


def get_args():
    '''
    Gets the arguments from the command line.
    '''
    parser = argparse.ArgumentParser("Run inference on an input video")
    # -- Create the descriptions for the commands
    m_desc = "The location of the model XML file"
    i_desc = "The location of the input file"
    d_desc = "The device name, if not 'CPU'"
    c_desc = "The color of the bounding boxes to draw; RED, GREEN or BLUE"
    ct_desc = "The confidence threshold to use with the bounding boxes"

    # -- Add required and optional groups
    parser._action_groups.pop()
    required = parser.add_argument_group('required arguments')
    optional = parser.add_argument_group('optional arguments')
    optional.add_argument("-c", help=c_desc, default='BLUE')
    optional.add_argument("-ct", help=ct_desc, default=0.5)

    # -- Create the arguments
    required.add_argument("-m", help=m_desc, required=True)
    optional.add_argument("-i", help=i_desc, default=INPUT_STREAM)
    optional.add_argument("-d", help=d_desc, default='CPU')
    args = parser.parse_args()

    return args

def convert_color(color_string):
    '''
    Get the BGR value of the desired bounding box color.
    Defaults to Blue if an invalid color is given.
    '''
    colors = {"BLUE": (255,0,0), "GREEN": (0,255,0), "RED": (0,0,255)}
    out_color = colors.get(color_string)
    if out_color:
        return out_color
    else:
        return colors['BLUE']
        
def draw_boxes(frame, result, args, width, height):
    '''
    Draw bounding boxes onto the frame.
    '''
    for box in result[0][0]: # Output shape is 1x1x100x7
        conf = box[2]
        if conf >= args.ct:
            xmin = int(box[3] * width)
            ymin = int(box[4] * height)
            xmax = int(box[5] * width)
            ymax = int(box[6] * height)
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), args.c, 2)
    return frame

def infer_on_video(args):
    # Initialize the Inference Engine
    plugin = Network()
    
    # Load the network model into the IE
    plugin.load_model(args.m, args.d, CPU_EXTENSION)
    net_input_shape = plugin.get_input_shape()
    
    # Get and open video capture
    cap = cv2.VideoCapture(args.i)
    cap.open(args.i)

    # Grab the shape of the input 
    width = int(cap.get(3))
    height = int(cap.get(4))

    # Create a video writer for the output video
    # The second argument should be `cv2.VideoWriter_fourcc('M','J','P','G')`
    # on Mac, and `0x00000021` on Linux
    out = cv2.VideoWriter('out.mp4', 0x00000021, 30, (width,height))
    
    # Process frames until the video ends, or process is exited
    while cap.isOpened():
        # Read the next frame
        flag, frame = cap.read()
        if not flag:
            break
        key_pressed = cv2.waitKey(60)

        # Pre-process the frame
        p_frame = cv2.resize(frame, (net_input_shape[3], net_input_shape[2]))
        p_frame = p_frame.transpose((2,0,1))
        p_frame = p_frame.reshape(1, *p_frame.shape)

        # Perform inference on the frame
        start_time = time.time()
        plugin.async_inference(p_frame)
        fps = 1.0 / (time.time() - start_time)
        fps = format(fps, '.2f')
        fps_str = "fps: " + str(fps) + "    CPU: Intel i7-6820HQ"

        args.c = convert_color(args.c)
        args.ct = float(args.ct)
        
        # Get the output of inference
        if plugin.wait() == 0:
            result = plugin.extract_output()
            ### Update the frame to include detected bounding boxes
            frame = draw_boxes(frame, result, args, width, height)
            # Write out the frame
            #out.write(frame)
    
        #print("FPS: ", 1.0 / (time.time() - start_time))
        cv2.putText(frame, fps_str, (10,30), font, 1,(255,255,255),2)
        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        # Update the frame to include detected bounding boxes
        #frame = draw_boxes(frame, result, args, width, height)

        # Write out the frame
        out.write(frame)
        # Break if escape key pressed
        if key_pressed == 27:
            break

    # Release the out writer, capture, and destroy any OpenCV windows
    out.release()
    cap.release()
    cv2.destroyAllWindows()


def main():
    args = get_args()
    infer_on_video(args)


if __name__ == "__main__":
    main()
