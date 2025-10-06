### infer

```sh
$ sima-model-to-pipeline infer --help

 Usage: sima-model-to-pipeline infer [OPTIONS]

 Creates a pipeline for inference. Currently supported for model types available in --decode-type option.     Run this command for visualization on host:

 gst-launch-1.0 udpsrc port=7000 ! application/x-rtp,encoding-name=H264,payload=96 ! rtph264depay ! "video/x-h264,stream-format=byte-stream,alignment=au" ! avdec_h264 ! fpsdisplaysink

╭─ Options ──────────────────────────────────────────────────────────────╮
│ --device [davinci/modalix]   Type of board to use for compilation.     │
│                                 [default: davinci]                     │
│                                                                        │
│ * --device-ip TEXT            Provide device IP address.               │
│                                 [default: None] [required]             │
│                                                                        │
│ * --model TEXT                Model path to create and run pipeline.   │
│                                 [default: None] [required]             │
│                                                                        │
│ * --rtsp-src TEXT             RTSP stream URL.                         │
│                                 [default: None] [required]             │
│                                                                        │
│ * --host-ip TEXT              Host IP address for visualization.       │
│                                 [default: None] [required]             │
│                                                                        │
│ --host-port TEXT              Port for visualization.                  │
│                                 [default: 7000]                        │
│                                                                        │
│ --labels-file TEXT            Path to labels file. If not provided,    │
│                               default YOLO 80 classes will be used.    │
│                               Required for PePPi.                      │
│                                                                        │
│ --pipeline-name TEXT          Name of the pipeline. Required for PePPi.│
│                                 [default: MyPipeline]                  │
│                                                                        │
│ --help                        Show this message and exit.              │
╰──────────────────────────────────────────────-─────────────────────────╯

```