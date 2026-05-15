# SimaaiBoxDecode

Generic plugin for postprocessing of model output in SiMa pipelines.

## Table of Contents

- [SimaaiBoxDecode](#pluginapplicationpipeline-name)
  - [Table of Contents](#table-of-contents)
  - [Introduction](#introduction)
  - [Features](#features)
  - [Getting Started](#getting-started)
    - [Prerequisites](#prerequisites)
  - [Usage](#usage)
  - [Configuration](#configuration)

## Features

Supports detesselation, dequantization and filtering of MLA output. For proper description and support of models please, refer to simaaiboxdecode library readme in a65-apps repository

## Getting Started

To build the plugin use:
```BASH
source environment-setup-cortexa65-poky-linux
mkdir build && cd build && cmake .. && make
```

### Prerequisites

- Yocto SDK installed

## Usage

```BASH
simaaisrc location='/data/mla-1.out' node-name='simaai_process_mla' \
! simaaiboxdecode config='/data/config.json' name='overlay' \
! fakesink
```

## Configuration

Configuration is splited to 3 blocks:

- GenerixBoxDecode library parameters (to get list of parameters, please, refer to library readme)
- Caps block (for detailed description of caps block, please, refer to Caps Negotiation Library README)
- Plugin pararmeters. They are common for every aggregator template base plugin:
  - `node_name` – name of current node (used as name of output buffer);
  - `memory` – block, that describes output memory options:
    - `cpu` – CPU where this plugin will be executed. Affects **only** memory allocation;
    - `next_cpu` – CPU where next plugin will be executed. Affects **only** memory allocation.
  - `system` – block, that holds system setings of plugin:
    - `out_buf_queue` – size of BufferPool;
    - `dump_data` – dump output buffers in `/tmp/{name-of-the-object}-{frame_id}.out`
  - `buffers` – block, that describes input/output buffers:
    - `input` – array of objects that holds name of input buffer and it's size
    - `output` – block, that holds output buffer size

Actual examples of all 3 blocks can be found in [example config.json](configuration-file-example).

## Buffer Size Calculation:
#### For BBOX only Models:
```
Buffer Size = 4 + (topK x 24) → 4: 4 bytes header for BBOXES, topK: 24 (Default), 24: BBOX size, so total: 580
```
#### For Segmentation Model: 
```
Buffer Size = 4 + topK x (24 + 160 x 160) → 4: 4 bytes header for BBOXES, topK: 20 , 24: BBOX size, 160: Mask size, so total: 512,484
```

## Buffer Size Calculation:
#### For BBOX only Models:
```
Buffer Size = 4 + (topK x 24) → 4: 4 bytes header for BBOXES, topK: 24 (Default), 24: BBOX size, so total: 580
```
#### For Segmentation Model: 
```
Buffer Size = 4 + topK x (24 + 160 x 160) → 4: 4 bytes header for BBOXES, topK: 20 , 24: BBOX size, 160: Mask size, so total: 512,484
```

## Configuration file example

```JSON
{
  "version": 0.1,
  "node_name": "simaai_boxdecode",
  "memory": {
    "cpu": 0,
    "next_cpu": 1
  },
  "system": {
    "out_buf_queue": 1,
    "debug": 0,
    "dump_data": 0
  },
  "buffers": {
    "input": [
      {
        "name": "simaai_process_mla",
        "size": 16000
      }
    ],
    "output": {
      "size": 580
    }
  },
  "decode_type" : "detr",
  "topk" : 24,
  "original_width": 1280,
  "original_height": 720,
  "model_width" : 640,
  "model_height" : 480,
  "num_classes" : 92,
  "detection_threshold" : 0.9,
  "nms_iou_threshold" : 0,
  "num_in_tensor": 2,
  "input_width": [
    100,
    100
  ],
  "input_height": [
    1,
    1
  ],
  "input_depth": [
    92,
    4
  ],
  "slice_width": [
    50,
    100
  ],
  "slice_height": [
    1,
    1
  ],
  "slice_depth": [
    92,
    4
  ],
  "dq_scale": [
    6.950103398907683,
    512.0
  ],
  "dq_zp": [
    37,
    -127357
  ],
  "data_type": [
    "INT8",
    "INT32"
  ],
  "caps": {
    "sink_pads": [
      {
        "media_type": "application/vnd.simaai.tensor",
        "params": [
          {
            "name": "format",
            "type": "string",
            "values": "MLA",
            "json_field": null
          },
          {
            "name": "data_type",
            "type": "string",
            "values": "(INT8, INT16, INT32), (INT8, INT16, INT32)",
            "json_field": "data_type"
          },
          {
            "name": "width",
            "type": "int",
            "values": "(1 - 4096), (1 - 4096)",
            "json_field": "input_width"
          },
          {
            "name": "height",
            "type": "int",
            "values": "(1 - 4096), (1 - 4096)",
            "json_field": "input_height"
          },
          {
            "name": "depth",
            "type": "int",
            "values": "(1 - 4096), (1 - 4096)",
            "json_field": "input_depth"
          },
          {
            "name": "slice_width",
            "type": "int",
            "values": "(1 - 4096), (1 - 4096)",
            "json_field": "slice_width"
          },
          {
            "name": "slice_height",
            "type": "int",
            "values": "(1 - 4096), (1 - 4096)",
            "json_field": "slice_height"
          },
          {
            "name": "slice_depth",
            "type": "int",
            "values": "(1 - 4096), (1 - 4096)",
            "json_field": "slice_depth"
          }
        ]
      }
    ],
    "src_pads": [
      {
        "media_type": "application/vnd.simaai.tensor",
        "params": [
          {
            "name": "format",
            "type": "string",
            "values": "BBOX",
            "json_field": null
          }
        ]
      }
    ]
  }
}
```