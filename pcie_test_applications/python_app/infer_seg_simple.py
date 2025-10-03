import argparse
import sys
import os
import logging
from time import time, sleep
import json
import cv2
import numpy as np

import base64
from utils_seg import intf

from imutils.video import VideoStream
from imutils.video import FPS


def unload_model(devintf):
    # Sleep for 5 seconds
    sleep(2)
    # Unload model
    print(f"Killing (unloading) the pipeline...")
    ret = devintf.unload_model(devintf.model)
    if ret != 0:
        print("Failed to unload model")
    print(f"Killed (unloaded) the pipeline.")

    # Disconnect the device
    print(f"Disconnecting the PCIe device: {devintf.soc_device}")
    ret = devintf.disconnect(devintf.device)
    if ret != 0:
        print(f"Failed to disconnect PCIe slot: {devintf.soc_device}")
    print(f"Disconnected the PCIe device: {devintf.soc_device}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog='test_simahostpy.py',
        description='Example python script to test connection, deployment of a pipeline, killing the pipeline and disconnection from the PCIe device.',
        epilog='')

    parser.add_argument('-f', help='Path to the mpk file to test with.', required=True, default=None)
    parser.add_argument('-i', help='Path to the input directory.', default='./../test_images')
    parser.add_argument('-o', help='Path to the output directory.', default='./output')
    parser.add_argument('--show_output', action='store_true', help='Display the output')
    args = parser.parse_args()

    print(f"Testing with packaged project.mpk file...")
    print(f"input model path: {args.f}")
    print(f"show output: {args.show_output}")

    if os.path.exists(args.f):
        devintf = intf(mpk_package=args.f)
    else:
        print(f'mpk file:{args.f} does not exists')
        sys.exit(1)

    try:
        if not os.path.isdir(args.i):    
            print(f"Found input {args.i} is not a directory. Starting video stream... inference.")
            if isinstance(args.i, str) and args.i.isdigit():
                args.i = int(args.i)

            vs = VideoStream(args.i).start()
            # start the FPS timer
            fps = FPS().start()
            # loop over frames from the video file stream
            while True:
                frame = vs.read()
                if frame is None:
                    print("frame is None, exiting...")
                    break

                results = devintf.run_inference_image(frame, annotate=True)
                annotated_frame = results["image"]

                if args.show_output is True:
                # show the frame and update the FPS counter
                    cv2.imshow("Frame", annotated_frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break

                fps.update()
            # stop the timer and display FPS information
            fps.stop()
            print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
            print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

            # do a bit of cleanup
            if args.show_output:
                cv2.destroyAllWindows()

        elif os.path.isdir(args.i):
            print(f"Testing with user provided project.mpk file {args.f}...")
            # run(args.i, args.o, args.f)

            input_images_path = args.i
            output_images_path = args.o
            if not os.path.exists(output_images_path):
                os.makedirs(output_images_path)
            
            print(f"Reading images from {input_images_path}")
            images = [f for f in os.listdir(input_images_path) if f.lower().endswith(('.jpg', '.jpeg'))]
            images.sort()
            print("Total test images:", len(images))

            count = 0        
            total_start_inference = time()

            for image in images[:]:
                count += 1
                st_tm = time()
                image_path = os.path.join(input_images_path, image)
                frame = cv2.imread(image_path)
                in_img_name = os.path.basename(image_path)
                print(f"[DEBUG]: Processing input image {in_img_name} {frame.shape}.")
                results = devintf.run_inference_image(frame, annotate=True)
                output_image_path = os.path.join(output_images_path, in_img_name)
                print(f"[DEBUG]: Saving output image {in_img_name} {frame.shape}.")
                cv2.imwrite(output_image_path, results["image"])

        else:
            print(f"Invalid input path: {args.i}. Please provide a valid image file or directory containing images.")
            unload_model(devintf)
            # sys.exit(1)
                    
    except Exception as e:
        print("Failed with ERROR: ' %s' ", str(e))
        print(f"Killing (unloading) the pipeline...")
        unload_model(devintf)
        sys.exit(1)
        
    unload_model(devintf)
    print(f"Completed Testing.")