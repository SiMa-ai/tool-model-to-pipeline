
# Steps to Create MPK using SDK Docker

## Start the SDK Docker
```bash
cd ~/workspace/sdk/sdk_1.7/1.7.0_Palette_SDK_master_B219/sima-cli/
./start.py
```

## Install the Python Package inside the SDK docker
```bash
cd /home/docker/sima-cli/sima_mpk_generator_with_CPP_sample
pip3 install sima_model_to_pipeline-sdk-1.7.tar.gz
```


# Yolo Object-Detection 
## Detection models supported:

     yolov8n,yolov8m,yolov8l
     yolov9t,s,m,c
     yolov10n,s,m,b,x
     yolo11n,s,m,l

## PCIe pipeline cmd for Detection:

```
sima-model-to-pipeline \
     model-to-pipeline  \
     --model-path yolov8n.onnx \
     --model-name yolov8 \
     --pipeline-name yolov8n_pipeline

sima-model-to-pipeline \
     model-to-pipeline  \
     --model-path yolov8n.onnx \
     --model-name yolov8 \
     --pipeline-name yolov8n_pipeline \
     --step pipelinecreate
```

#### Here you can replace yolv8n with any of the model above, 
#### Examples:
```
Yolov9:
sima-model-to-pipeline \
     model-to-pipeline \
     --model-path yolov9s.onnx \
     --model-name yolov9 \
     --pipeline-name yolov9s_pipeline

Yolov10:
sima-model-to-pipeline \
     model-to-pipeline \
     --model-path yolov10s.onnx \
     --model-name yolov10 \
     --pipeline-name yolov10s_pipeline

Yolo11:
sima-model-to-pipeline \
     model-to-pipeline \
     --model-path yolo11s.onnx \
     --model-name yolo11 \
     --pipeline-name yolo11s_pipeline

```

## Install the required packages:

### Install the required packages:
```bash
sudo apt update
sudo apt install rpm2cpio -y
sudo apt-get install python3-opencv

# Python packages:
pip3 install opencv-python
pip3 install imutils
```

### Install SiMa PCIe Drivers

```bash
cd ~/workspace/sdk/sdk_1.6/release_1658
sudo ./sima_pcie_host_pkg.sh
```

## Python test app:
```
cd python_app

# RTSP Stream as input
python3 infer_detect.py -f yolo11/yolov11s_pipeline/project.mpk -i rtsp://192.168.134.90/axis-media/media.amp --show_output

# Video file as input
python3 infer_detect.py -f yolo11/yolov11s_pipeline/project.mpk -i ~/Videos/people_walking.mp4 --show_output

# Images folder as input
python3 infer_detect.py -f yolo11/yolov11s_pipeline/project.mpk -i ./input_images/ -o ./output/
```

## CPP test application

### Setup enviroment and build application:
```
cd cpp_detection_app
mkdir build
cd build
cmake ..
make
```
### Run inference:

#### For Images inference:
```bash
./run_inference project.mpk ./../test_images
```
#### For IP-Cameras inference:
```bash
./run_inference project.mpk rtsp://192.168.134.90/axis-media/media.amp
```
#### For Video file inference:
```bash
./run_inference project.mpk video_file.mp4
```
#### For Web camera or USB cameras
```bash
./run_inference project.mpk 0
```

## EtherNet pipeline for Detection:

```
sima-model-to-pipeline \
    model-to-pipeline  \
    --model-path yolov8n.onnx \
    --model-name yolov8 \
    --pipeline-name yolov8n_ethernet_pipeline \
    --host-ip 172.16.1.20 \
    --host-port 5000 \
    --input-width 1280 \
    --input-height 720 \
    --rtsp-src "rtsp://192.168.132.125:8554/mystream1"
```

### Deploy the created mpk manually
```
mpk device connect -t sima@{mlsoc-ip}
mpk deploy -f yolov8n_ethernet_pipeline/project.mpk
```

### Run the GST-CLI on the host to see the output
```
$ GST_DEBUG=0 gst-launch-1.0 udpsrc port=5000 ! application/x-rtp,encoding-name=H264,payload=96 ! rtph264depay ! 'video/x-h264,stream-format=byte-stream,alignment=au' !  avdec_h264  ! fpsdisplaysink sync=false
```


# Yolo Instance Segmentation:

### ** PCIe pipeline cmd for Segmentation: (Segmentation is only supported in the PCIe mode and Python script)

## Segmentation models supported:

     yolov8n-seg,yolov8m-seg,yolov8l-seg 
     yolo11n-seg, yolo11s-seg, yolo11m-seg, yolo11l-seg, yolo11x-seg

## PCIe pipeline cmd for Segmentation:

```
sima-model-to-pipeline \
     model-to-pipeline  \
     --model-path yolov8n-seg.onnx \
     --model-name yolov8-seg \
     --pipeline-name yolov8n_seg_pipeline \
     --no-box-decode \
     --model-type segmentation 
```

#### Here you can replace yolv8n with any of the model above, 
#### Examples:

#####  Yolov9:
```
sima-model-to-pipeline \
     model-to-pipeline \
     --model-path yolov9s-seg.onnx \
     --model-name yolov9 \
     --pipeline-name yolov9s_seg_pipeline \
     --no-box-decode \
     --model-type segmentation 
```
##### Yolo11:
```
sima-model-to-pipeline \
     model-to-pipeline \
     --model-path yolo11s-seg.onnx \
     --model-name yolo11 \
     --pipeline-name yolo11s_seg_pipeline \
     --no-box-decode \
     --model-type segmentation 

```

### For Custom model example:

```
sima-model-to-pipeline \
     model-to-pipeline  \
     --model-path customer_yolov8m_640.onnx \
     --model-name yolov8 \
     --pipeline-name yolov8n_pipeline \
     --labels-file labels.txt \
     --calibration-data-path ./images_for_calibration/ \
     --num-classes 6

```



### Python test app:

```
# RTSP Stream as input
python3 infer_seg_simple.py -f yolo11_seg/yolov11s_seg_pipeline/project.mpk -i rtsp://192.168.134.90/axis-media/media.amp --show_output

# Video file as input
python3 infer_seg_simple.py -f yolo11_seg/yolov11s_seg_pipeline/project.mpk -i ~/Videos/people_walking.mp4 --show_output

# Images folder as input
python3 infer_seg_simple.py -f yolo11_seg/yolov11s_seg_pipeline/project.mpk -i ./input -o ./output
```





Yolo-X

Download the Yolo-X model seperately:
wget: https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_s.onnx

```
sima-model-to-pipeline \
     model-to-pipeline  \
     --model-path yolox_s.onnx \
     --model-name yolox \
     --pipeline-name yoloxs_pipeline \
     --model-type segmentation \
     --no-box-decode 
```
#### *Python application work-in-progress
