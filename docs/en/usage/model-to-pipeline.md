## model-to-pipeline

```sh
$ sima-model-to-pipeline model-to-pipeline --help

 Usage: sima-model-to-pipeline model-to-pipeline [OPTIONS]
```

Most commonly, the command is run with a sample [YAML input file](samples/yolov9c.yaml):

```sh
$ sima-model-to-pipeline model-to-pipeline --config-yaml samples/yolov9c.yaml
```

* This will run and do graph surgery on vanilla `<mymodel>.onnx` and create `<mymodel>_mod.onnx`.
* Using the model post surgery, the tool will compile the model to generate `.elf` file.
* Using this generated `.elf` file, the tool will create a mini pipeline using `mpk project create`

This tool also supports command line argument in case user wants a finer-grained control.

| Option | Type / Choices | Description | Default |
|--------|----------------|-------------|---------|
| `--model-path` | TEXT | Path to the model file to use | None |
| `--model-name` | yolov8 / yolov9 / yolov8-seg | Name of the model | None |
| `--post-surgery-model-path` | TEXT | Path to the model file after surgery | None |
| `--model-type` | object-detection / image-classification / segmentation | Type of the model | object-detection |
| `--input-width` | INTEGER | Input width of the pipeline | None |
| `--input-height` | INTEGER | Input height of the pipeline | None |
| `--compilation-result-dir` | TEXT | Directory where compiled model will be dumped | result |
| `--compiler` | yolo | Name of compiler | yolo |
| `--calibration-data-path` | TEXT | Path to the calibration dataset | None |
| `--calibration-samples-count` | INTEGER | Max number of calibration samples | None |
| `--arm-only / --no-arm-only` | FLAG | Compile for ARM architecture only | no-arm-only |
| `--act-asym / --no-act-asym` | FLAG | Use asymmetric activation quantization | no-act-asym |
| `--act-per-ch / --no-act-per-ch` | FLAG | Use per-channel activation quantization | no-act-per-ch |
| `--act-bf16 / --no-act-bf16` | FLAG | Keep precision as bf16 for activation quantization | no-act-bf16 |
| `--act-nbits` | 4 / 8 / 16 | Activation quantization bit precision | 8 |
| `--wt-asym / --no-wt-asym` | FLAG | Use asymmetric weight quantization | no-wt-asym |
| `--wt-per-ch / --no-wt-per-ch` | FLAG | Use per-channel weight quantization | no-wt-per-ch |
| `--wt-bf16 / --no-wt-bf16` | FLAG | Keep precision as bf16 for weight quantization | no-wt-bf16 |
| `--wt-nbits` | 4 / 8 / 16 | Weight quantization bit precision | 8 |
| `--bias-correction` | none / iterative / regular | Bias correction type | none |
| `--ceq / --no-ceq` | FLAG | Enable channel equalization | no-ceq |
| `--smooth-quant / --no-smooth-quant` | FLAG | Enable smooth quantization | no-smooth-quant |
| `--compress / --no-compress` | FLAG | Compress model during compilation | no-compress |
| `--mode` | sima / tflite | Requantization mode | sima |
| `--calibration-type` | min_max / moving_average / entropy / percentile / mse | Calibration algorithm | mse |
| `--batch-size` | INTEGER | Batch size for compilation | 1 |
| `--calibration-ds-extn` | TEXT | File extension for calibration images | jpg |
| `--device-type` | davinci / modalix / both | Target device type | davinci |
| `--input-resource` | TEXT | Path to input image or video | None |
| `--pipeline-name` | TEXT | Final name of the pipeline | None |
| `--config-yaml` | TEXT | Provide configuration from YAML file | None |
| `--no-box-decode / --no-no-box-decode` | FLAG | Use `detessdequant` instead of `simaaiboxdecode` | false |
| `--rtsp-src` | TEXT | RTSP stream (RTSP pipeline only) | None |
| `--host-ip` | TEXT | Host IP for RTSP streaming | None |
| `--host-port` | TEXT | Host port for RTSP streaming | None |
| `--detection-threshold` | FLOAT | Detection confidence threshold | 0.4 |
| `--nms-iou-threshold` | FLOAT | IoU threshold for NMS | 0.4 |
| `--topk` | INTEGER | Maximum number of detections | 25 |
| `--labels-file` | TEXT | Path to labels file | None |
| `--num-classes` | INTEGER | Number of output classes | 80 |
| `--step` | TEXT | Run only the specified step (others skipped) | None |
| `--help` | FLAG | Show help message and exit | - |


## Progress and Summary

While running, the timer clock keeps showing the elapsed time and a progress bar indicating the process is still running. This gives the summary at the end of execution as below.

A monitor server is started automatically providing a more comphrensive interface for user to view the logs as the tool processes through the model. Copy and paste the server into a browser on the host side to access the monitor server.

```sh
Monitor server running at http://172.18.0.2:5000?input=yolov8m.yaml
Starying Flask server on port 5000 ...
✅ Completed : setup               : 0.10 sec
✅ Completed : downloadmodel       : 7.24 sec
✅ Completed : surgery             : 1.52 sec
✅ Completed : downloadcalib       : 0.10 sec
✅ Completed : compile             : 156.12 sec
✅ Completed : pipelinecreate      : 0.80 sec
✅ Completed : mpkcreate           : 74.96 sec


      SiMa.ai Model to Pipeline Summary
╭────────────────┬──────────────┬────────────╮
│   Step Name    │ Elapsed Time │   Status   │
├────────────────┼──────────────┼────────────┤
│ downloadmodel  │     PASS     │  7.25 sec  │
├────────────────┼──────────────┼────────────┤
│    surgery     │     PASS     │  1.52 sec  │
├────────────────┼──────────────┼────────────┤
│ downloadcalib  │     PASS     │  0.11 sec  │
├────────────────┼──────────────┼────────────┤
│    compile     │     PASS     │ 156.13 sec │
├────────────────┼──────────────┼────────────┤
│ pipelinecreate │     PASS     │  0.81 sec  │
├────────────────┼──────────────┼────────────┤
│   mpkcreate    │     PASS     │ 74.97 sec  │
╰────────────────┴──────────────┴────────────╯
                   Summary

```