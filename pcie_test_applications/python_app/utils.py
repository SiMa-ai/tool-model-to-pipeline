import os
import shutil
import json
import argparse
import sys
import os
import logging
from time import time, sleep
import json
import cv2
import numpy as np
from simahostpy import *
import struct
import importlib
from typing import Union
import shutil

import base64

def convert_np(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.generic,)):
        return obj.item()
    elif isinstance(obj, dict):
        return {k: convert_np(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_np(v) for v in obj]
    else:
        return obj

def get_model_info(mpk_package):
    """
    Get model information from the MPK package.
    
    :param mpk_package: Path to the MPK package.
    :return: Dictionary containing model information.
    """
    print(f"Loading the model from the file path: {mpk_package}")
    # Create temporary directory
    os.system("rm -rf /tmp/sima_tmp")
    os.system("mkdir /tmp/sima_tmp")

    # Extract MPK file 
    shutil.copyfile(mpk_package, "/tmp/sima_tmp/project.mpk")

    # Extract contents
    os.system("cd /tmp/sima_tmp/; unzip /tmp/sima_tmp/project.mpk")
    os.system("cd /tmp/sima_tmp/*/resources/; rpm2cpio installer.rpm | cpio -idmv")
    os.system("cp /tmp/sima_tmp/*/resources/data/simaai/applications/*/etc/* /tmp/sima_tmp/")

    # Read JSON file
    with open("/tmp/sima_tmp/boxdecoder.json") as f:
        mpk_json = json.load(f)

    # Get formatted labels
    labels = []
    if "labels" in mpk_json:
        labels = [label for label in mpk_json["labels"]]

    # Get dimensions
    height = 0
    width = 0 
    topk = 0
    if "original_width" in mpk_json and "original_height" in mpk_json:
        height = mpk_json["original_height"]
        width = mpk_json["original_width"]
        topk = mpk_json["topk"]

    # Cleanup
    os.system("rm -rf /tmp/sima_tmp")

    return {"height": height, "width": width, "topk": topk, "labels": labels}

def setup_logger(filepath, sys_log_level=logging.INFO, console_log_level=logging.INFO):
	if os.path.isfile(filepath):
		os.remove(filepath)
	# Create a logger
	logger = logging.getLogger()
	logger.setLevel(sys_log_level)  # Set the logging level
	
	# Create handlers
	console_handler = logging.StreamHandler()
	console_handler.setLevel(console_log_level)  # Set the logging level for the handler
	file_handler = logging.FileHandler(filepath)
	file_handler.setLevel(sys_log_level)  # Set the logging level for the handler

	# Create a formatter and set it for both handlers
	formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
	# console_handler.setFormatter(formatter)
	file_handler.setFormatter(formatter)

	# Add handlers to the logger
	logger.addHandler(console_handler)
	logger.addHandler(file_handler)
	return logger




logger = setup_logger("pciehost.log")

package = 'simahostpy'
simaaihostpy_implementation = importlib.import_module(package)

class BaseEnum():
    @classmethod
    def values(cls):
        return [member.value for member in cls]

class SiMaErrorCode(BaseEnum):
    SUCCESS = 0
    FAILURE = 1
    INVALID_INPUT = 2
    OVERFLOW = 3
    RETRY = 4

error_code_to_infer_state = {
    "0": SiMaErrorCode.SUCCESS,
    "1": SiMaErrorCode.FAILURE,
    "2": SiMaErrorCode.INVALID_INPUT,
    "3": SiMaErrorCode.OVERFLOW,
    "4": SiMaErrorCode.RETRY,
}

class Constants:
    DEVICE_NAME           = "simapcie"
    FRAME_HEIGHT          = 640 # Original frame height
    FRAME_WIDTH           = 640 # Original frame width
    INPUT_SHAPE           = [FRAME_WIDTH,FRAME_HEIGHT,3]
    OUTPUT_SHAPE	      = [580]
    IN_BUFFER_SIZE        = int(FRAME_HEIGHT * FRAME_WIDTH * 3)
    OUT_BUFFER_SIZE       = int(580)*4
    DUMP_OUTPUT           = True # Dump json output
    REMOVE_PROCESSED      = False # Remove processed images from input_dir (This must be True for production env)
    MIN_TIME_OUT          = 0.5 # Time out for RESET (Not in use)
    MAX_TIME_OUT          = 3 # Time out for re-deploy in Seconds
    INFER_TIMEOUT         = 5 # Maximum allowable response time
    SOC_WAIT_TIME         = 10 # Sleep time after each deploy
    MAX_REDEPLOY_INTERVAL = 60 * 2 # Redeploy interval in Seconds
    EN_REDEPLOY           = False # Enable/Disable Redeploy
    EN_RESET              = True
    REDEPLOY_RETRY        = 5 # Number of consecutive retries for REDEPLOY
    DEVICE_TIMEOUT        = 60 # PCIE timeout
    DEVICE_QUEUE          = 8 # PCIE buffer queue
    FRAME_COUNT           = 5 # Allowed frame count before each re-deploy
    ANNOTATE              = False # Annotate PMs and dump it to file
    conf_threshold        = 0.2
    iou_threshold         = 0.1
    class_names           = None
    H, W                  = FRAME_WIDTH, FRAME_HEIGHT
    color                 = (0, 255, 0)  # Green color for bounding boxes

class intf:
    def __init__(self, mpk_package):
        self.dev_name = None
        self.device_queue = None
        self.device_timeout = None
        self.host_helper = None
        self.dev_ptr = None
        self.guids = []
        self.model_ref = None
        self.meta_data = []

        self.model = None
        self.mpk_package = mpk_package
        self.devintf = None
        self.device = None
        self.soc_device = None
        
        self.host_helper = simaaihostpy_implementation.HostHelper()
        if (self.host_helper == None):
            raise Exception("Couldn't get the device inference interface")

        self.guids = self.host_helper.enumerate_device()
        if (len(self.guids) <= 0):
            return
        
        print(f"Number of devices available: {len(self.guids)}")

        if mpk_package is not None:
            print(f"Using MPK package: {self.mpk_package}")
            if not os.path.exists(mpk_package):
                raise FileNotFoundError(f"MPK package '{mpk_package}' does not exist.")
            self.mpk_package = mpk_package
            print(f"Using MPK package: {self.mpk_package}")

            """
            Load model from MPK package.
            :param mpk_package: Path to the MPK package.
            :return: Model reference.
            """
            model_info = get_model_info(mpk_package)
            print(f"Model Information: {model_info}")
            Constants.FRAME_HEIGHT = model_info["height"]
            Constants.FRAME_WIDTH = model_info["width"]
            Constants.IN_BUFFER_SIZE = int(model_info["width"] * model_info["height"] * 3)
            
            # Constants.OUT_BUFFER_SIZE = int(model_info["topk"]*24 )  # Assuming 6 integers per box (x1, y1, w, h, conf, classid)
            Constants.class_names = model_info["labels"]
            Constants.INPUT_SHAPE = [Constants.FRAME_WIDTH, Constants.FRAME_HEIGHT, 3]

            # Constants.OUTPUT_SHAPE = [4+Constants.OUT_BUFFER_SIZE // 4]  # Adjusted for number of boxes and attributes
            print(f"Updated Constants: IN_BUFFER_SIZE={Constants.IN_BUFFER_SIZE}, OUT_BUFFER_SIZE={Constants.OUT_BUFFER_SIZE}, INPUT_SHAPE={Constants.INPUT_SHAPE}, OUTPUT_SHAPE={Constants.OUTPUT_SHAPE}")

            # exit()

            soc_devices = simaaihostpy_implementation.HostHelper().enumerate_device()
            
            print("Number of soc devices available: ", len(soc_devices))
            print("---------------------------------")
            for idx, device in enumerate(soc_devices, start=1):
                print(f"{idx}. {device}")
            print("---------------------------------")
            
            for soc_device in soc_devices:
                # choice = input(f"Would you like to continue on {soc_device.strip()}? (y/n): ").strip().lower()
                # if choice != 'y':
                #     continue
                self.dev_name = Constants.DEVICE_NAME
                self.device_queue = Constants.DEVICE_QUEUE
                self.device_timeout = Constants.DEVICE_TIMEOUT
                self.soc_device = soc_device.strip()

                in_out_sz = int(Constants.IN_BUFFER_SIZE * 0.25)
                out_sz = int(Constants.OUT_BUFFER_SIZE * 0.25)
                model_hdl_org = {
                "in_tensors": 1,
                "out_tensors": 1,
                "out_batch_sz": 1,
                "in_batch_sz": 1,
                "in_shape": [in_out_sz],
                "out_shape": [out_sz],
                }

                print(f"Connecting to PCIe device: {soc_device}")
                self.device = self.connect(soc_device)
                print(f"Connected to PCIe device: {soc_device}")

                # Populate Model
                model_hdl = {}
                model_hdl['numInputTensors'] = 1
                model_hdl['numOutputTensors'] = 1
                model_hdl['outputBatchSize'] = 1
                model_hdl['inputBatchSize'] = 1
                model_hdl['inputShape'] = Constants.INPUT_SHAPE
                model_hdl['outputShape'] = Constants.OUTPUT_SHAPE
                model_hdl['qid'] = 0

                model_hdl_org.update(model_hdl)

                # Deploy model
                print(f"Deploying {mpk_package} to PCIe device: {soc_device}")

                try:
                    self.model = self.load_model(self.device,[[in_out_sz]],[[out_sz]],['A', 'B'], mpk_package, model_hdl=model_hdl_org)
                    sleep(Constants.SOC_WAIT_TIME)
                except Exception as e:
                    print("Initialize soc for %s failed with ERROR: ' %s' ", soc_device, str(e))
                    sys.exit(0)

                print(f"Successfully deployed (loaded) the pipeline.")

    def connect(self, guid, queue_entries = 0, queue_depth = 0):
        if guid is None:
            raise ValueError("Guid cannot be NULL, please pass a valid guid")

        if self.host_helper is None:
            raise Exception(" The inference interface is not initialized")
        
        self.dev_ptr = self.host_helper.open(guid)
        if self.dev_ptr is None:
            raise Exception("Unable to connect to the device")

        self.host_helper.print_slot_number(self.dev_ptr)
        self.host_helper.update_device_defaults(self.dev_ptr,
                                                queue_entries,
                                                queue_depth)
        return self.dev_ptr

    def prep_tens(self, in_shape_list, out_shape_list, meta_data):
        self.host_helper.prepare_tensors(in_shape_list,
                                         out_shape_list, 0)
        self.host_helper.set_metadata(meta_data)
        
    def load_model(self, device,
                   in_shape_list, out_shape_list,
                   metadata,
                   model_path = None, model_hdl:dict = None):

        if (device != self.dev_ptr):
            raise Exception("Device mismatch")
        
        if ((in_shape_list is None) or (out_shape_list is None)):
            raise ValueError('Shapes of in and out tensors cannot be None')

        self.host_helper.prepare_tensors(in_shape_list,
                                         out_shape_list,0)

        self.host_helper.set_metadata(metadata)
        self.meta_data = metadata
        self.host_helper.set_queue(device, self.device_queue)
        self.host_helper.set_request_timeout(device, self.device_timeout)
        
        if((model_hdl is not None) and (model_path is not None)):
            model_def = self.host_helper.set_model_definition(
                model_hdl["in_tensors"],
                model_hdl["out_tensors"],
                model_hdl["in_batch_sz"],
                model_hdl["out_batch_sz"],
                model_hdl["in_shape"],
                model_hdl["out_shape"],
            )
            
            self.model_ref = self.host_helper.load(device,
                                                   model_path,
                                                   model_def)
            if (self.model_ref is None):
                raise Exception(f'Unable to load model_hdl to the device {self.model_ref}')
            return self.model_ref 
        

        if (model_hdl is not None):
            model_def = self.host_helper.set_model_definition(
                model_hdl["in_tensors"],
                model_hdl["out_tensors"],
                model_hdl["in_batch_sz"],
                model_hdl["out_batch_sz"],
                model_hdl["in_shape"],
                model_hdl["out_shape"],
            )
            
            self.model_ref = self.host_helper.load(device,
                                                   model_def)
        else:
            self.model_ref = self.host_helper.load(device,
                                                   model_path)
            
        if (self.model_ref is None):
            raise Exception(f'Unable to load model_hdl to the device {self.model_ref}')

        return self.model_ref

    def preprocess(self, input_image):
        target_size = (Constants.FRAME_WIDTH, Constants.FRAME_HEIGHT)
        print('target_size:',target_size)
        width_ratio = target_size[0] / input_image.shape[1]
        height_ratio = target_size[1] / input_image.shape[0]
        
        if width_ratio < height_ratio:
            new_height = int(input_image.shape[0] * width_ratio)
            resized_image = cv2.resize(input_image, (target_size[0], new_height), cv2.INTER_AREA)
            top_padding = (target_size[1] - new_height) // 2
            bottom_padding = target_size[1] - new_height - top_padding
            resized_image = cv2.copyMakeBorder(resized_image, top_padding, bottom_padding, 0, 0, 
                                            cv2.BORDER_CONSTANT, value=(0, 0, 0))
        else:
            new_width = int(input_image.shape[1] * height_ratio)
            resized_image = cv2.resize(input_image, (new_width, target_size[1]), cv2.INTER_AREA)
            left_padding = (target_size[0] - new_width) // 2
            right_padding = target_size[0] - new_width - left_padding
            resized_image = cv2.copyMakeBorder(resized_image, 0, 0, left_padding, right_padding, 
                                            cv2.BORDER_CONSTANT, value=(0, 0, 0))
        
        # Convert BGR to RGB first (before normalization)
        resized_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB)
        #normalized_image = resized_image / 255. #((input_image / 255. - mean) / std).astype(np.float32)
        #normalized_image = normalized_image.astype(np.float32)
    
        return resized_image

    def __update__(self, in_data: Union[np.ndarray, bytes]):
        if in_data is None:
            raise Exception('Input numpy array cannot be None')

        self.host_helper.memcpy(in_data, 0)

    def __get_tensor__(self, out_data: Union[np.ndarray, bytes]):
        if out_data is None:
            raise Exception('Input numpy array cannot be None')
        print("[DEBUG]: Copy to host buffer...")
        self.host_helper.memcpy_tonp(out_data, 0)

    def unload_model(self, model):
        if (self.model_ref != model):
            raise Exception("Model refernce mismatch")
        ret = self.host_helper.unload(model)
        return error_code_to_infer_state[str(ret.value)] 

    def disconnect(self,device):
        if (device != self.dev_ptr):
            raise Exception("Device handle mismatch")
        ret = self.host_helper.close(device)
        return error_code_to_infer_state[str(ret.value)]

    
    def run_infer(self, model, frame, count):
        print("Allocating buffer of size ", Constants.OUT_BUFFER_SIZE)
        out_tensor = np.zeros([int(Constants.OUT_BUFFER_SIZE)], dtype=np.uint8)
        logger.info("Sending frame to soc ")
        self.host_helper.memcpy(frame, 0)
        out = self.host_helper.run_inference(model)
        if(out != SiMaErrorCode.SUCCESS):
            print(f"{count}: Run inference failure. STATUS {out}")
            return out
        self.host_helper.memcpy_tonp(out_tensor, 0)
        if (out is None):
            raise Exception(f'Unable to run the inference {out}')
        else:
            print(f"[DEBUG]: RUN INFERENCE SUCCESSFUL. STATUS {out}")
        return out, out_tensor

    def run_inference_images(self, input_images_path, output_images_path):
        """
        Load model from MPK package.
        :param mpk_package: Path to the MPK package.
        :return: Model reference.
        """
        print(f"[DEBUG] Run Inference.")
        print(f"Reading images from {input_images_path}")
        images = [f for f in os.listdir(input_images_path) if f.lower().endswith(('.jpg', '.jpeg'))]
        images.sort()
        print("Total test images:", len(images))

        count = 0        
        total_start_inference = time()
        try:
            
            for image in images[:]:
                count += 1
                st_tm = time()*1000
                image_path = os.path.join(input_images_path, image)
                logger.info("Sending %s to soc ", image_path)
                frame = cv2.imread(image_path)
                in_img_name = os.path.basename(image_path)
                print(f"[DEBUG]: Processing input image {frame.shape}.")

                orig_h, orig_w = frame.shape[:2]
                input_h, input_w = Constants.FRAME_HEIGHT, Constants.FRAME_WIDTH  # e.g., 640, 640
                # Determine the scale and padding
                r = min(input_w / orig_w, input_h / orig_h)
                new_w, new_h = int(orig_w * r), int(orig_h * r)
                pad_w = (input_w - new_w) // 2
                pad_h = (input_h - new_h) // 2


                frame_rgb = self.preprocess(frame)
                print(f"[DEBUG]: Processing input image after preprocess {frame_rgb.shape}.")
                out_tensor = np.zeros([int(Constants.OUT_BUFFER_SIZE)], dtype=np.uint8)
                out, out_tensor = self.run_infer(self.model, frame_rgb, count)

                frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
                if(out != SiMaErrorCode.SUCCESS):
                    print(f"{count}: Run inference failure. Status is {out}")
                    continue
                # Convert to 32-bit integers (little-endian)
                int_data = out_tensor.view('<u4')  # or '<i4' for signed integers
                offset = 0
                num_boxes =  int_data[offset]
                results = []
                boxes,conf,class_id = [], [], []
                for i in range(num_boxes):
                    x1 = int_data[offset+  1]
                    y1 = int_data[offset + 2]
                    w = int_data[offset + 3]
                    h = int_data[offset + 4]
                    x2 = x1 + w
                    y2 = y1 + h

                    

                    conf = struct.unpack('f', struct.pack('I', int_data[offset + 5]))[0]
                    classid = int_data[offset + 6]
                    results.append({
                        "box": [x1, y1, x2, y2],
                        "confidence": conf,
                        "class_id": classid
                    })

                    print(f"Process output. output tensor shape is {int_data.shape}, number of boxes {num_boxes}, x1 = {x1}, y1 = {y1}, x2 = {x2}, y2 = {y2}, conf = {conf}, classid = {classid} ")
                    offset += 6
                    
                    # Remove padding, then scale
                    x1 = (x1 - pad_w) / r
                    y1 = (y1 - pad_h) / r
                    x2 = (x2 - pad_w) / r
                    y2 = (y2 - pad_h) / r

                    # Clip to original image size
                    x1 = int(max(0, min(x1, orig_w))) #orig_shape[1]))
                    y1 = int(max(0, min(y1, orig_h))) # orig_shape[0]))
                    x2 = int(max(0, min(x2, orig_w))) #orig_shape[1]))
                    y2 = int(max(0, min(y2, orig_h))) #orig_shape[0]))
                    print(f"Adjusted box coordinates: x1={x1}, y1={y1}, x2={x2}, y2={y2}")

                    # Draw rectangle
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                    # Draw text
                    cv2.putText(frame, Constants.class_names[classid], (x1, y1 - 10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

                output_path = os.path.join(output_images_path, in_img_name)
                print(f"[DEBUG]: Writing output image to {output_path}.")
                cv2.imwrite(output_path, frame)
                
                print(f"results: {results}")
                end_tm = time()*1000
                print(f"{count}: Inference run time  {end_tm -st_tm}.")
            
        except Exception as e:
            print("Error in soc for %s failed with ERROR: ' %s' ", self.soc_device, str(e))
            print(f"Killing (unloading) the pipeline...")
            ret = self.unload_model(self.model)
            if ret != 0:
                print("Failed to unload model")
            print(f"Killed (unloaded) the pipeline.")

            # Disconnect the device
            # print(f"Disconnecting the PCIe device: {soc_device}")
            ret = self.disconnect(self.device)
            if ret != 0:
                print(f"Failed to disconnect PCIe slot")
            print(f"Disconnected the PCIe device")

            sys.exit(0)


        print(f"[DEBUG]: Run Inference is complete. Total execution time:  {time()-total_start_inference}")
            
        # Sleep for 5 seconds
        sleep(5)

        # Unload model
        print(f"Killing (unloading) the pipeline...")
        ret = self.unload_model(self.model)
        if ret != 0:
            print("Failed to unload model")
        print(f"Killed (unloaded) the pipeline.")

        # Disconnect the device
        print(f"Disconnecting the PCIe device: {self.soc_device}")
        ret = self.disconnect(self.device)
        if ret != 0:
            print(f"Failed to disconnect PCIe slot: {self.soc_device}")
        print(f"Disconnected the PCIe device: {self.soc_device}")

    def run_inference_image(self, frame, annotate=False):
        """
        Load model from MPK package.
        :param mpk_package: Path to the MPK package.
        :return: Model reference.
        """
        print(f"[DEBUG] Run Inference.")
        count = 0        
        total_start_inference = time()
        try:
            st_tm = time()*1000
            orig_h, orig_w = frame.shape[:2]
            input_h, input_w = Constants.FRAME_HEIGHT, Constants.FRAME_WIDTH  # e.g., 640, 640
            # Determine the scale and padding
            r = min(input_w / orig_w, input_h / orig_h)
            new_w, new_h = int(orig_w * r), int(orig_h * r)
            pad_w = (input_w - new_w) // 2
            pad_h = (input_h - new_h) // 2

            frame_rgb = self.preprocess(frame)
            print(f"[DEBUG]: Processing input image after preprocess {frame_rgb.shape}.")
            out_tensor = np.zeros([int(Constants.OUT_BUFFER_SIZE)], dtype=np.uint8)
            out, out_tensor = self.run_infer(self.model, frame_rgb, count)
            if(out != SiMaErrorCode.SUCCESS):
                print(f"{count}: Run inference failure. Status is {out}")
                # continue
            # Convert to 32-bit integers (little-endian)
            int_data = out_tensor.view('<u4')  # or '<i4' for signed integers
            offset = 0
            num_boxes =  int_data[offset]
            results = {"results": [], "image": None}
            boxes,conf,class_id = [], [], []
            for i in range(num_boxes):
                x1 = int_data[offset+  1]
                y1 = int_data[offset + 2]
                w = int_data[offset + 3]
                h = int_data[offset + 4]
                x2 = x1 + w
                y2 = y1 + h
                conf = struct.unpack('f', struct.pack('I', int_data[offset + 5]))[0]
                classid = int_data[offset + 6]
                # Remove padding, then scale
                x1 = (x1 - pad_w) / r
                y1 = (y1 - pad_h) / r
                x2 = (x2 - pad_w) / r
                y2 = (y2 - pad_h) / r

                # Clip to original image size
                x1 = int(max(0, min(x1, orig_w))) #orig_shape[1]))
                y1 = int(max(0, min(y1, orig_h))) # orig_shape[0]))
                x2 = int(max(0, min(x2, orig_w))) #orig_shape[1]))
                y2 = int(max(0, min(y2, orig_h))) #orig_shape[0]))
                print(f"Adjusted box coordinates: x1={x1}, y1={y1}, x2={x2}, y2={y2}")

                results["results"].append({
                    "box": [x1, y1, x2, y2],
                    "confidence": conf,
                    "class_id": classid
                })

                print(f"Process output. output tensor shape is {int_data.shape}, number of boxes {num_boxes}, x1 = {x1}, y1 = {y1}, x2 = {x2}, y2 = {y2}, conf = {conf}, classid = {classid} ")
                offset += 6
                
                if annotate:
                    # Draw rectangle
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                    # Draw text
                    cv2.putText(frame, Constants.class_names[classid], (x1, y1 - 10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            if annotate:
                # _, buffer = cv2.imencode(".jpg", frame)
                # img_b64 = base64.b64encode(buffer).decode()
                # results["image"] = img_b64
                results["image"] = frame


            print(f"results: {results['results']}")
            end_tm = time()*1000
            print(f"{count}: Inference run time  {end_tm -st_tm}.")

            results["results"] = convert_np(results["results"])
            return results
            
        except Exception as e:
            print("Initialize soc for %s failed with ERROR: ' %s' ", self.soc_device, str(e))
            print(f"Killing (unloading) the pipeline...")
            ret = self.unload_model(self.model)
            if ret != 0:
                print("Failed to unload model")
            print(f"Killed (unloaded) the pipeline.")

            # Disconnect the device
            # print(f"Disconnecting the PCIe device: {soc_device}")
            ret = self.disconnect(self.device)
            if ret != 0:
                print(f"Failed to disconnect PCIe slot")
            print(f"Disconnected the PCIe device")

            sys.exit(0)


        print(f"[DEBUG]: Run Inference is complete. Total execution time:  {time()-total_start_inference}")
        
        return None