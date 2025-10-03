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
import random
import math


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
    model_height          = FRAME_HEIGHT # Model input height
    model_width           = FRAME_WIDTH # Model input width
    INPUT_SHAPE           = [FRAME_WIDTH,FRAME_HEIGHT,3]
    W, H                  = FRAME_WIDTH, FRAME_HEIGHT
    class_names           = None
    # Read class names from file
    # label_file_path       = os.path.join(os.path.dirname(__file__), 'labels.txt')
    # with open(label_file_path, 'r') as f:
    #     class_names = [line.strip() for line in f if line.strip()]
    
    num_classes           = 80 
    num_masks             = 32 # Number of masks
    OUTPUT_SHAPE	      = [
                                (1, W//8, H//8, 4),
                                (1, W//16, H//16, 4),
                                (1, W//32, H//32, 4),
                                (1, W//8, H//8, num_classes),
                                (1, W//16, H//16, num_classes),
                                (1, W//32, H//32, num_classes),
                                (1, W//8, H//8, num_masks), 
                                (1, W//16, H//16, num_masks), 
                                (1, W//32, H//32, num_masks), 
                                (1, W//4, H//4, num_masks)
                            ]
    IN_BUFFER_SIZE        = int(FRAME_HEIGHT * FRAME_WIDTH * 3)
    OUT_BUFFER_SIZE       = (sum(np.prod(OUT_SHAPE) for OUT_SHAPE in OUTPUT_SHAPE)) * 4
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
    
    buffer_sizes          = [
                                (1, W//8, H//8, 4),
                                (1, W//16, H//16, 4),
                                (1, W//32, H//32, 4),
                                (1, W//8, H//8, num_classes),
                                (1, W//16, H//16, num_classes),
                                (1, W//32, H//32, num_classes),
                                (1, W//8, H//8, 32), 
                                (1, W//16, H//16, 32), 
                                (1, W//32, H//32, 32), 
                                (1, W//4, H//4, 32)
                            ]
    color                 = (0, 255, 0)  # Green color for bounding boxes
    mask_color            = [(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)) for _ in range(num_classes)] 

class intf:
    def __init__(self, dev_name, device_queue, device_timeout):
        self.dev_name = dev_name
        self.device_queue = device_queue
        self.device_timeout = device_timeout
        self.host_helper = None
        self.dev_ptr = None
        self.guids = []
        self.model_ref = None
        self.meta_data = []

        self.model = None
        self.mpk_package = None
        self.devintf = None
        self.device = None
        self.soc_device = None
        
        self.host_helper = simaaihostpy_implementation.HostHelper()
        if (self.host_helper == None):
            raise Exception("Couldn't get the device inference interface")

        self.guids = self.host_helper.enumerate_device()
        if (len(self.guids) <= 0):
            return

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
            resized_image = cv2.resize(input_image, (target_size[0], new_height))
            top_padding = (target_size[1] - new_height) // 2
            bottom_padding = target_size[1] - new_height - top_padding
            resized_image = cv2.copyMakeBorder(resized_image, top_padding, bottom_padding, 0, 0, 
                                            cv2.BORDER_CONSTANT, value=(0, 0, 0))
        else:
            new_width = int(input_image.shape[1] * height_ratio)
            resized_image = cv2.resize(input_image, (new_width, target_size[1]))
            left_padding = (target_size[0] - new_width) // 2
            right_padding = target_size[0] - new_width - left_padding
            resized_image = cv2.copyMakeBorder(resized_image, 0, 0, left_padding, right_padding, 
                                            cv2.BORDER_CONSTANT, value=(0, 0, 0))
        
        # Convert BGR to RGB first (before normalization)
        resized_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB)

        # Normalize the image will happend in the SoC
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

    @staticmethod
    def rescale_boxes(boxes, input_shape, image_shape):
        # Rescale boxes to original image dimensions
        input_shape = np.array([input_shape[1], input_shape[0], input_shape[1], input_shape[0]])
        boxes = np.divide(boxes, input_shape, dtype=np.float32)
        boxes *= np.array([image_shape[1], image_shape[0], image_shape[1], image_shape[0]])

        return boxes

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def xywh2xyxy(self, x):
        # Convert bounding box (x, y, w, h) to bounding box (x1, y1, x2, y2)
        y = np.copy(x)
        y[..., 0] = x[..., 0] - x[..., 2] / 2
        y[..., 1] = x[..., 1] - x[..., 3] / 2
        y[..., 2] = x[..., 0] + x[..., 2] / 2
        y[..., 3] = x[..., 1] + x[..., 3] / 2
        return y

    def extract_boxes(self, box_predictions, 
                    # input_shape=(Constants.FRAME_HEIGHT, Constants.FRAME_WIDTH), 
                    # image_shape=(Constants.FRAME_HEIGHT, Constants.FRAME_WIDTH)
                    ):
        # Extract boxes from predictions
        boxes = box_predictions[:, :4]
        print("boxes shape in extract_boxes", boxes.shape)

        # boxes = self.rescale_boxes(boxes,
        #                             input_shape,
        #                             image_shape)

        # Convert boxes to xyxy format
        boxes = self.xywh2xyxy(boxes)

        # Check the boxes are within the image
        boxes[:, 0] = np.clip(boxes[:, 0], 0, Constants.model_width)
        boxes[:, 1] = np.clip(boxes[:, 1], 0, Constants.model_height)
        boxes[:, 2] = np.clip(boxes[:, 2], 0, Constants.model_width)
        boxes[:, 3] = np.clip(boxes[:, 3], 0, Constants.model_height)

        return boxes

    def reshape_sima_output(self, output, num_classes):
        # BBOX: (1, 80, 80, 4), (1, 40, 40, 4), (1, 20, 20, 4)
        # Prob: (1, 80, 80, 80), (1, 40, 40, 80), (1, 20, 20, 80)
        #print("outpout shape in sima reshape", output, num_classes)
        if len(output) == 1:
            return output
        
        pred_bbox = []
        for k in range(0, 3):
            bbox = output[k]
            pred_bbox.append(bbox.reshape(1, -1, 4))
        pred_bbox = np.concatenate(pred_bbox, axis=1)

        # print("pred_bbox shape in sima reshape", pred_bbox.shape)
        
        pred_prob = []
        for k in range(3, 6):
            p = output[k]
            pred_prob.append(p.reshape(1, -1, num_classes))
        pred_prob = np.concatenate(pred_prob, axis=1)

        # print("pred_prob shape in sima reshape", pred_prob.shape)

        pred_coef = []
        for k in range(6,9):
            coef = output[k]
            pred_coef.append(coef.reshape(1, -1, 32))

        pred_coef = np.concatenate(pred_coef, axis=1)
        # print("pred_coef shape in sima reshape", pred_coef.shape)

        pred = np.concatenate([pred_bbox, pred_prob, pred_coef], 
                                axis=2).transpose(0, 2, 1)


        return pred, output[9].transpose(0, 3, 1, 2)

    def compute_iou(self, box, boxes):
        # Compute xmin, ymin, xmax, ymax for both boxes
        xmin = np.maximum(box[0], boxes[:, 0])
        ymin = np.maximum(box[1], boxes[:, 1])
        xmax = np.minimum(box[2], boxes[:, 2])
        ymax = np.minimum(box[3], boxes[:, 3])

        # Compute intersection area
        intersection_area = np.maximum(0, xmax - xmin) * np.maximum(0, ymax - ymin)

        # Compute union area
        box_area = (box[2] - box[0]) * (box[3] - box[1])
        boxes_area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        union_area = box_area + boxes_area - intersection_area

        # Compute IoU
        iou = intersection_area / union_area

        return iou

    def nms(self, boxes, scores, iou_threshold):
        # Sort by score
        sorted_indices = np.argsort(scores)[::-1]

        keep_boxes = []
        while sorted_indices.size > 0:
            # Pick the last box
            box_id = sorted_indices[0]
            keep_boxes.append(box_id)

            # Compute IoU of the picked box with the rest
            ious = self.compute_iou(boxes[box_id, :], boxes[sorted_indices[1:], :])

            # Remove boxes with IoU over the threshold
            keep_indices = np.where(ious < Constants.iou_threshold)[0]

            # print(keep_indices.shape, sorted_indices.shape)
            sorted_indices = sorted_indices[keep_indices + 1]

        return keep_boxes

    def process_box_output(self, box_output):
        
        predictions = np.squeeze(box_output).T    
        num_classes = box_output.shape[1] - Constants.num_masks - 4

        # Filter out object confidence scores below threshold
        scores = np.max(predictions[:, 4:4+num_classes], axis=1)
        predictions = predictions[scores > Constants.conf_threshold, :]
        scores = scores[scores > Constants.conf_threshold]

        if len(scores) == 0:
            return [], [], [], np.array([])

        box_predictions = predictions[..., :num_classes+4]
        mask_predictions = predictions[..., num_classes+4:]

        # Get the class with the highest confidence
        class_ids = np.argmax(box_predictions[:, 4:], axis=1)

        # Get bounding boxes for each object
        boxes = self.extract_boxes(box_predictions) #, input_shape=img_array.shape[:2])

        # Apply non-maxima suppression to suppress weak, overlapping bounding boxes
        indices = self.nms(boxes, scores, Constants.iou_threshold)
        print("indices after nms:", indices)

        return boxes[indices], scores[indices], class_ids[indices], mask_predictions[indices]


    def process_mask_output(self, boxes, mask_predictions, mask_output):

        if mask_predictions.shape[0] == 0:
            return []

        mask_output = np.squeeze(mask_output)

        # Calculate the mask maps for each box
        num_mask, mask_height, mask_width = mask_output.shape  # CHW
        masks = self.sigmoid(mask_predictions @ mask_output.reshape((num_mask, -1)))
        masks = masks.reshape((-1, mask_height, mask_width))

        # Downscale the boxes to match the mask size
        scale_boxes = self.rescale_boxes(boxes,
                                    (Constants.model_height, Constants.model_width),
                                    (mask_height, mask_width))

        # For every box/mask pair, get the mask map
        mask_maps = np.zeros((len(scale_boxes), Constants.model_height, Constants.model_width))
        blur_size = (int(Constants.model_width / mask_width), int(Constants.model_height / mask_height))
        for i in range(len(scale_boxes)):

            scale_x1 = int(math.floor(scale_boxes[i][0]))
            scale_y1 = int(math.floor(scale_boxes[i][1]))
            scale_x2 = int(math.ceil(scale_boxes[i][2]))
            scale_y2 = int(math.ceil(scale_boxes[i][3]))

            x1 = int(math.floor(boxes[i][0]))
            y1 = int(math.floor(boxes[i][1]))
            x2 = int(math.ceil(boxes[i][2]))
            y2 = int(math.ceil(boxes[i][3]))

            scale_crop_mask = masks[i][scale_y1:scale_y2, scale_x1:scale_x2]
            crop_mask = cv2.resize(scale_crop_mask,
                                (x2 - x1, y2 - y1),
                                interpolation=cv2.INTER_CUBIC)

            crop_mask = cv2.blur(crop_mask, blur_size)

            crop_mask = (crop_mask > 0.5).astype(np.uint8)
            mask_maps[i, y1:y2, x1:x2] = crop_mask

        return mask_maps

    def scale_boxes_back(self, boxes, original_shape, resized_shape, pad, scale):
        # boxes: (N, 4) format [x1, y1, x2, y2]
        # Subtract pad and divide by scale
        if len(boxes) != 0:
            boxes[:, [0, 2]] -= pad[0]  # x padding
            boxes[:, [1, 3]] -= pad[1]  # y padding
            boxes /= scale
            boxes = boxes.clip(min=0)  # ensure boxes are valid

        return boxes

    def letterbox_reverse_params(self, original_shape, resized_shape):
        r = min(resized_shape[0] / original_shape[0], resized_shape[1] / original_shape[1])
        new_unpad = (int(round(original_shape[1] * r)), int(round(original_shape[0] * r)))
        pad_w = resized_shape[1] - new_unpad[0]
        pad_h = resized_shape[0] - new_unpad[1]
        pad = (pad_w // 2, pad_h // 2)
        return r, pad



    def scale_mask_back(self, mask_maps, original_shape, resized_shape, pad, scale):
        # Remove padding
        rescaled_masks = []
        if len(mask_maps) == 0:
            return None
        for mask in mask_maps:
            y1, y2 = int(pad[1]), int(resized_shape[0] - pad[1])
            x1, x2 = int(pad[0]), int(resized_shape[1] - pad[0])
            mask_unpadded = mask[y1:y2, x1:x2]

            # Resize back to original shape
            original_h, original_w = original_shape
            mask_rescaled = cv2.resize(mask_unpadded, (original_w, original_h), interpolation=cv2.INTER_LINEAR)
            rescaled_masks.append(mask_rescaled)
        return rescaled_masks

    def process_output(self, output_tensor):
        binary_data = output_tensor.tobytes()
        output_tensor = np.frombuffer(binary_data, dtype=np.float32)
        reshaped_data = []
        offset = 0
        for shape in Constants.OUTPUT_SHAPE:
            size = np.prod(shape)
            reshaped_data.append(output_tensor[offset:offset+size].reshape(shape))
            # print('reshaped_data.shape:',reshaped_data[-1].shape)
            offset += size

        outputs, mask = self.reshape_sima_output(reshaped_data, num_classes=len(Constants.class_names))

        boxes, scores, class_ids, mask_pred = self.process_box_output(outputs)

        mask_maps = self.process_mask_output(boxes, mask_pred, mask)
        return boxes, scores, class_ids, mask_maps

    def draw_masks(self, image, boxes, class_ids, mask_alpha=0.5, mask_maps=None):
        mask_img = image.copy()

        # Draw bounding boxes and labels of detections
        for i, (box, class_id) in enumerate(zip(boxes, class_ids)):
            color = Constants.mask_color[class_id]

            x1, y1, x2, y2 = box.astype(int)

            # Draw fill mask image
            if mask_maps is None:
                cv2.rectangle(mask_img, (x1, y1), (x2, y2), color, -1)
            else:
                crop_mask = mask_maps[i][y1:y2, x1:x2, np.newaxis]
                crop_mask_img = mask_img[y1:y2, x1:x2]
                crop_mask_img = crop_mask_img * (1 - crop_mask) + crop_mask * color
                mask_img[y1:y2, x1:x2] = crop_mask_img

        return cv2.addWeighted(mask_img, 0.5, image, 1 - 0.5, 0)

    def draw_detections(self, image, boxes, scores, class_ids, mask_alpha=0.5, mask_maps=None):

        img_height, img_width = image.shape[:2]
        size = min([img_height, img_width]) * 0.0006
        text_thickness = int(min([img_height, img_width]) * 0.001)

        mask_img = self.draw_masks(image, boxes, class_ids, mask_alpha, mask_maps)
        
        # Draw bounding boxes and labels of detections
        for box, score, class_id in zip(boxes, scores, class_ids):
            color = Constants.mask_color[class_id]

            x1, y1, x2, y2 = box.astype(int)

            # Draw rectangle
            cv2.rectangle(mask_img, (x1, y1), (x2, y2), color, 2)

            label = Constants.class_names[class_id]
            caption = f'{label} {int(score * 100)}%'
            (tw, th), _ = cv2.getTextSize(text=caption, fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                                        fontScale=size, thickness=text_thickness)
            th = int(th * 1.2)

            cv2.rectangle(mask_img, (x1, y1),
                        (x1 + tw, y1 - th), color, -1)

            cv2.putText(mask_img, caption, (x1, y1),
                        cv2.FONT_HERSHEY_SIMPLEX, size, (255, 255, 255), text_thickness, cv2.LINE_AA)

        return mask_img

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
    with open("/tmp/sima_tmp/mpk.json") as f:
        mpk_json = json.load(f)

    # Get formatted labels
    labels = []
    if "labels" in mpk_json["extra_automation_params"]:
        labels = [label for label in mpk_json["extra_automation_params"]["labels"]]

    # Get dimensions
    height = 0
    width = 0 
    topk = 0
    # if "input_width" in mpk_json and "input_width" in mpk_json["extra_automation_params"]:
    # height = mpk_json["extra_automation_params"]["input_height"]
    # width = mpk_json["extra_automation_params"]["input_width"]
    # topk = mpk_json["extra_automation_params"]["topk"]
    # model_width = mpk_json["extra_automation_params"]["model_input_width"]
    # model_height = mpk_json["extra_automation_params"]["model_input_height"]
    # Cleanup
    os.system("rm -rf /tmp/sima_tmp")
    # print(f"Model information: height={height}, width={width}, topk={topk}, labels={labels}")
    model_details = mpk_json["extra_automation_params"]
    # {
    #     "height": height,
    #     "width": width,
    #     "topk": topk,
    #     "labels": labels,
    #     "model_width": model_width,
    #     "model_height": model_height
    # }
    return model_details


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



def run(args, input_images_path, output_images_path, mpk_package="project.mpk"):

    model_info = get_model_info(mpk_package)
    print(f"Model Information: {model_info}")
    Constants.FRAME_HEIGHT = model_info["input_height"]
    Constants.FRAME_WIDTH = model_info["input_width"]
    Constants.model_height = model_info["model_input_height"]
    Constants.model_width = model_info["model_input_width"]
    Constants.IN_BUFFER_SIZE = int(model_info["input_height"] * model_info["input_width"] * 3)
    
    # Constants.OUT_BUFFER_SIZE = int(model_info["topk"]*24 )  # Assuming 6 integers per box (x1, y1, w, h, conf, classid)
    Constants.class_names = model_info["labels"]
    W, H                  = Constants.model_width, Constants.model_height
    Constants.INPUT_SHAPE = [W, H, 3]
    num_classes = len(Constants.class_names)
    num_masks = 32
    Constants.OUTPUT_SHAPE = [
                                (1, W//8, H//8, 4),
                                (1, W//16, H//16, 4),
                                (1, W//32, H//32, 4),
                                (1, W//8, H//8, num_classes),
                                (1, W//16, H//16, num_classes),
                                (1, W//32, H//32, num_classes),
                                (1, W//8, H//8, num_masks), 
                                (1, W//16, H//16, num_masks), 
                                (1, W//32, H//32, num_masks), 
                                (1, W//4, H//4, num_masks)
                            ]
    Constants.IN_BUFFER_SIZE        = int(Constants.FRAME_HEIGHT * Constants.FRAME_WIDTH * 3)
    Constants.OUT_BUFFER_SIZE       = (sum(np.prod(OUT_SHAPE) for OUT_SHAPE in Constants.OUTPUT_SHAPE)) * 4

    # Constants.OUTPUT_SHAPE = [4+Constants.OUT_BUFFER_SIZE // 4]  # Adjusted for number of boxes and attributes
    print(f"Updated Constants: IN_BUFFER_SIZE={Constants.IN_BUFFER_SIZE}, OUT_BUFFER_SIZE={Constants.OUT_BUFFER_SIZE}, INPUT_SHAPE={Constants.INPUT_SHAPE}, OUTPUT_SHAPE={Constants.OUTPUT_SHAPE}")

    soc_devices = simaaihostpy_implementation.HostHelper().enumerate_device()
    
    print("Number of soc devices available: ", len(soc_devices))
    print("---------------------------------")
    for idx, device in enumerate(soc_devices, start=1):
        print(f"{idx}. {device}")
    print("---------------------------------")
    
    for soc_device in soc_devices:
        choice = input(f"Would you like to continue on {soc_device.strip()}? (y/n): ").strip().lower()
        if choice != 'y':
            continue
        devintf = intf(Constants.DEVICE_NAME, Constants.DEVICE_QUEUE, Constants.DEVICE_TIMEOUT)

        devintf.dev_name = Constants.DEVICE_NAME
        devintf.device_queue = Constants.DEVICE_QUEUE
        devintf.device_timeout = Constants.DEVICE_TIMEOUT
        devintf.soc_device = soc_device.strip()
        devintf.mpk_package = mpk_package

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
        device = devintf.connect(soc_device)
        devintf.device = device
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
            model = devintf.load_model(device,[[in_out_sz]],[[out_sz]],['A', 'B'], mpk_package, model_hdl=model_hdl_org)
            devintf.model = model
            sleep(Constants.SOC_WAIT_TIME)

            print(f"Successfully deployed (loaded) the pipeline.")
            
            if not os.path.isdir(input_images_path):    
                from imutils.video import VideoStream
                from imutils.video import FPS

                print(f"Found input {input_images_path} is not a directory. Starting video stream... inference.")
                if isinstance(input_images_path, str) and input_images_path.isdigit():
                    input_images_path = int(input_images_path)

                vs = VideoStream(input_images_path).start()
                fps = FPS().start()

                # loop over frames from the video file stream
                while True:
                    frame = vs.read()
                    if frame is None:
                        print("frame is None, exiting...")
                        break

                    print(f"[DEBUG]: Processing input image {frame.shape}.")
                    if (Constants.FRAME_WIDTH, Constants.FRAME_HEIGHT) != (Constants.model_width, Constants.model_height):
                        frame = cv2.resize(frame, (Constants.FRAME_WIDTH, Constants.FRAME_HEIGHT))
                        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    else:
                        frame_rgb = devintf.preprocess(frame)

                    print(f"[DEBUG]: Processing input image after preprocess {frame_rgb.shape}.")
                    start_inf = time()
                    out_tensor = np.zeros([int(Constants.OUT_BUFFER_SIZE)], dtype=np.uint8)
                    out, out_tensor = devintf.run_infer(model, frame_rgb, count=0)
                    time_taken_inf = (time() - start_inf)*1000
                    print(f"[Debug]: Inference time taken: {time_taken_inf}")

                    if(out != SiMaErrorCode.SUCCESS):
                        print(f"{count}: Run inference failure. Status is {out}")
                        continue

                    boxes, scores, class_ids, mask_maps = devintf.process_output(out_tensor)

                    scale, pad = devintf.letterbox_reverse_params(
                        original_shape=frame.shape[:2],
                        resized_shape=(Constants.model_height, Constants.model_width)
                    )

                    rescaled_boxes = devintf.scale_boxes_back(boxes,
                                                    resized_shape=(Constants.model_height, Constants.model_width),
                                                    original_shape=frame.shape[:2],
                                                    pad=pad,
                                                    scale=scale)
                    rescaled_mask_maps = devintf.scale_mask_back(mask_maps,
                                                        resized_shape=(Constants.model_height, Constants.model_width),
                                                        original_shape=frame.shape[:2],
                                                        pad=pad,
                                                        scale=scale)
                    
                    print('boxes:', rescaled_boxes)
                    # print(f"[DEBUG]: Processed output boxes: {boxes}, scores: {scores}, class_ids: {class_ids}, mask_maps: {mask_maps if mask_maps is not None else 'None'}.")
                    annotated_frame =  devintf.draw_detections(frame, rescaled_boxes, scores, 
                                                            class_ids, 0.5, rescaled_mask_maps )
                    
                    if annotated_frame.dtype != np.uint8:
                        annotated_frame = annotated_frame.astype(np.uint8)

                    # results = devintf.run_inference_image(frame, annotate=True)
                    # annotated_frame = results["image"]

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
                    unload_model(devintf)
                    sys.exit(0)

                unload_model(devintf)
                sys.exit(0)

        except Exception as e:
            print("Failed with ERROR: ' %s' ", str(e))
            unload_model(devintf)
            sys.exit(0)


        # Validate input_images_path exists
        if not os.path.isdir(input_images_path):
            print(f"Error: input_images_path '{input_images_path}' does not exist.")
            sys.exit(1)

        # Handle output directory
        if os.path.exists(output_images_path):
            print(f"Deleting existing directory: {output_images_path}")
            shutil.rmtree(output_images_path)

        print(f"Creating directory: {output_images_path}")
        os.makedirs(output_images_path)



        # Run inference
        print(f"[DEBUG] Run Inference.")
        print("Reading images from {input_img_path}")
        images = [f for f in os.listdir(input_images_path) if f.lower().endswith(('.jpg', '.jpeg'))]
        images.sort()
        print("Total test images:", len(images))

        count = 0        
        total_start_inference = time()
        try:
            for image in images[:]:
                count += 1
                st_tm = time()
                image_path = os.path.join(input_images_path, image)
                frame = cv2.imread(image_path)
                
                in_img_name = os.path.basename(image_path)
                print(f"[DEBUG]: Processing input image {frame.shape}.")
                if (Constants.FRAME_WIDTH, Constants.FRAME_HEIGHT) != (Constants.model_width, Constants.model_height):
                    frame = cv2.resize(frame, (Constants.FRAME_WIDTH, Constants.FRAME_HEIGHT))
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                else:
                    frame_rgb = devintf.preprocess(frame)

                print(f"[DEBUG]: Processing input image after preprocess {frame_rgb.shape}.")
                out_tensor = np.zeros([int(Constants.OUT_BUFFER_SIZE)], dtype=np.uint8)
                out, out_tensor = devintf.run_infer(model, frame_rgb, count)
                if(out != SiMaErrorCode.SUCCESS):
                    print(f"{count}: Run inference failure. Status is {out}")
                    continue

                boxes, scores, class_ids, mask_maps = devintf.process_output(out_tensor)

                scale, pad = devintf.letterbox_reverse_params(
                                original_shape=frame.shape[:2],
                                resized_shape=(Constants.model_height, Constants.model_width)
                            )
                print(f"[DEBUG]: Scale: {scale}, Pad: {pad}.")
                print('boxes:',boxes)
                rescaled_boxes = devintf.scale_boxes_back(boxes,
                                                resized_shape=(Constants.model_height, Constants.model_width),
                                                original_shape=frame.shape[:2],
                                                pad=pad,
                                                scale=scale)
                rescaled_mask_maps = devintf.scale_mask_back(mask_maps,
                                                    resized_shape=(Constants.model_height, Constants.model_width),
                                                    original_shape=frame.shape[:2],
                                                    pad=pad,
                                                    scale=scale)
                
                print('boxes_:', rescaled_boxes)
                # print(f"[DEBUG]: Processed output boxes: {boxes}, scores: {scores}, class_ids: {class_ids}, mask_maps: {mask_maps if mask_maps is not None else 'None'}.")
                output_image =  devintf.draw_detections(frame, rescaled_boxes, scores, 
                                                        class_ids, 0.5, rescaled_mask_maps )
                
                if output_image.dtype != np.uint8:
                    output_image = output_image.astype(np.uint8)

                output_path = os.path.join(output_images_path, in_img_name)
                print(f"[DEBUG]: Writing output image to {output_path}.")
                cv2.imwrite(output_path, output_image)
                
                end_tm = time()
                print(f"{count}: Inference run time  {end_tm -st_tm}.")

            # Sleep for 2 seconds
            sleep(2)
            unload_model(devintf)
                
        except Exception as e:
            print("Failed with ERROR: ' %s' ", str(e))
            unload_model(devintf)
            sys.exit(0)


        print(f"[DEBUG]: Run Inference is complete. Total execution time:  {time()-total_start_inference}") 

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog='test_simahostpy.py',
        description='Example python script to test connection, deployment of a pipeline, killing the pipeline and disconnection from the PCIe device.',
        epilog='')

    parser.add_argument('-f', help='Path to the mpk file to test with.', required=True)
    parser.add_argument('-i', help='Path to the input directory.', default='./../test_images')
    parser.add_argument('-o', help='Path to the output directory.', default='./output_seg')
    parser.add_argument('--show_output', action='store_true', help='Display the output')
    args = parser.parse_args()

    
    print(f"Testing with user provided project.mpk file {args.f}...")
    run(args, args.i, args.o, args.f)
    
    print(f"Completed Inferencing.")