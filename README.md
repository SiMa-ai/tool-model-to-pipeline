# tool-model-to-pipeline

[![SDK Compatibility](https://img.shields.io/badge/SDK-1.7.0-blue.svg)](#)
[![Models](https://img.shields.io/badge/Supported-YOLO-green.svg)](#)
[![Python](https://img.shields.io/badge/Python-3.10%2B-orange.svg)](#)
[![License](https://img.shields.io/badge/License-MIT-lightgrey.svg)](LICENSE)

This tool converts FP32 YOLO models into working GStreamer pipelines for the SiMa platform.  

It is designed to be extensible, with support for additional models through pluggable [model-surgery](model_to_pipeline/surgeons) modules.


## Installation

Execute the command to install the package inside Palette SDK environment, be sure to upgrade ``sima-cli`` first following this [instruction](https://docs.sima.ai/pages/sima_cli/main.html).

```sh
$ sima-cli install gh:sima-ai/tool-model-to-pipeline
```

## Usage
This project provides three utilities:  

1. [model-to-pipeline](#1-model-to-pipeline) converter that transforms a model into a runnable pipeline.  
2. [get-fps](#2-get-fps) tool that extracts the MLA-only FPS number from a model’s `.tar.gz` file using `mla-rt`.  
3. [infer](#3-infer) tool that runs the deployed pipeline on the SiMa.ai device.


```sh
$ sima-model-to-pipeline --help

 Usage: sima-model-to-pipeline [OPTIONS] COMMAND [ARGS]...

╭─ Options ───────────────────────────────────────────────────────────────╮
│ --install-completion   Install completion for the current shell.        │
│ --show-completion      Show completion for the current shell, to copy   │
│                        it or customize the installation.                │
│ --help                 Show this message and exit.                      │
╰─────────────────────────────────────────────────────────────────────────╯

╭─ Commands ──────────────────────────────────────────────────────────────╮
│ model-to-pipeline   Convert a model into a working pipeline             │
│ get-fps             Find the MLA-only FPS using mla-rt                  │
│ infer               Create a pipeline for inference                     │
╰─────────────────────────────────────────────────────────────────────────╯
```

### 1. `model-to-pipeline`

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

<details>
<summary>Click to expand all command line options</summary>

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

</details>

#### Progress and Summary
While running, the timer clock keeps showing the elapsed time and a progress bar indicating the process is still running. This gives the summary at the end of execution as below.

```sh
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

### 2. `get-fps`

```sh
$ sima-model-to-pipeline get-fps --help

 Usage: sima-model-to-pipeline get-fps [OPTIONS]

 Finds the FPS number for MLA only using mla-rt

╭─ Options ──────────────────────────────────────────────────────────────╮
│ --device [davinci/modalix]   Type of the board to use for compilation. │
│                               [default: davinci]                       │
│                                                                        │
│ * --device-ip TEXT            Provide device IP address.               │
│                               [default: None] [required]               │
│                                                                        │
│ * --models TEXT               List of model paths (space-separated).   │
│                               [default: None] [required]               │
│                                                                        │
│ --help                        Show this message and exit.              │
╰─────────────────────────────────────────────────────────────────────-──╯

```

### 3. `infer`

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

## Logs

The logs of execution will be dumped under directory named `logs/`. Every step involved in the tool execution, a separate log file will be generated based on the step name.

<br>

Example logs are as follows.

```sh
logs/
├── compile.log
├── main.log
├── pipelinecreate.log
└── surgery.log
```

# Project Overview
This tool project has multiple files and directories supporting multiple functionalities. Here, we're using *__plugin-based architecture__* for adding new steps or surgery or compilation logics.

## `main.py`
This file is the main entry point of the tool

## `cli.py`
This file accepts the command line arguments to the tool and returns the `argparse.Namespace` instance

## `utils`
This directory contains plugins `loader`, process util for executing commands on the host using `subprocess` module. It also contains `onnx_helper.py` module used while doing graph surgery of the model and the logger module for keeping logs of all the steps involved.

## `compilers`
This directory contains all the compiler logic for different models

## `constants`
This is an internal usage directory where all constant values can be kept and used later in the project

## `pipeline`
This directory is for handling pipeline creating using compiled model's `tar.gz` file. It also modifies the plugins to include `genericboxdecode` and `overlay` along with their respective config files. <br>
This creates two `application.json` files in the same directory one for `rtsp` and other for `pcie` respectively.

## `surgeons`
This directory contains all surgeon scripts for doing model surgery.

## `steps`
This dorectory is for steps to be taken starting from model surgery through final project creation. This is the actual lifetime of the tool.

# Adding New
The tool is based on plugin architecture. If one wants to add a support for new models, it can be done easily without modifying existing code.

## Support for new model
We assume that below items are already available before adding a support for new model

* `.onnx` file

* Surgery script

* Compilation script

* Model's name in available choices for parameter `--model_name`

<br>

### Adding Model Surgeon
* Create a python script under `surgeons` directory

* Create a class and extend it with a surgeon's base class named `SurgeonBase` and add class variable named as `name`. Assign the model name __all in small letters__ to it so that it becomes easy to pick the correct surgeon while execution.

* Override a method named `do_surgery` with below signature

```python
class SurgeonMyModel(SurgeonBase):
    name = "mymodel" # This must be lower case

    def do_surgery(self, args: argparse.Namespace) -> str:
        """
        Perform the surgery on the MyModel model.
        This method should be overridden by subclasses.
        """
        # Surgery code for your model
```
* The body of your `do_surgery` implementation is the actual surgery script which you already have. You must need to do some modifications in your surgery script like correcting the paths if there are any.

### Adding Model Compiler

* Create a python script under `compilers` directory

* Create a class and extend it with a compiler's base class named `CompilerBase` and add class variable named as `name`. Assign the model name __all in small letters__ to it so that it becomes easy to pick the correct surgeon while execution.

* Override a method named `compile` with below signature

```python
class CompileMyModel(CompilerBase):
    """Compiler for MyModel models."""

    name = "mymodel" # This must be lower case

    def compile(self, args: argparse.Namespace):
        """
        Compile the MyModel model using SiMa's SDK.
        This method loads the ONNX model, extracts input and output shapes,
        and prepares the model for quantization and compilation.
        This method is designed to work with the SiMa SDK and assumes that
        the model is in ONNX format. It also supports calibration datasets
        for quantization.

        Args:
            args (argparse.Namespace): Commandline arguments

        Raises:
            RuntimeError: Raises runtime error
        """
        # Compilation code for your model
```

* The body of your `compile` implementation is the actual compilation script which you already have. You must need to do some modifications in your compilation script like correcting the paths if there are any.


### Update the choices for command line parameter `--model_name`
Update the `sima_model_to_pipeline/main.py` script to accept your new model. To do so, add your model's name under `metavar` putting a forward slash `/` after last option for `--model_name` parameter as below

```python
model_name: Optional[str] = typer.Option(
        None,
        help="Name of the model.",
        show_choices=True,
        case_sensitive=False,
        prompt=False,
        metavar="yolov8/yolov9",
    )
```

## Adding New Step
Follow this if you wish to add new step to workflow of the tool.

* Create a python script under `steps` directory

* Create a new step class and extend it with step's base class named `StepBase`. Add two class variables named `name` and `sequence` under it.
  * `name` is used to identify the step
  * `sequence` is used to sequence the steps for excution

* Override the `run` method in your custom step class

```python
class StepMyStep(StepBase):
    """Step for performing MyStep on the model.
    """

    name = "mystep"
    sequence = 4

    def run(self, args: argparse.Namespace) -> bool:
        """Run the mystep step.

        Args:
            args: Arguments passed to the step.

        Returns:
            bool: True if the mystep was successful, False otherwise.
        """
        # Step logic for your step
```

## Adding a new tool under `sima-model-to-pipeline`
To add a new tool altogether: <br>
1. Create a directory at root level of the project. For example, let's call it `my_new_tool`.<br>
2. Add a `__init__.py` under it so that that directory is considered as a module <br>
3. Create another python script under it which contains the main logic of your code. Make sure that you use `typer` module's cocept to make your sub-tool a part of main tool and accessible via it. You can use below template for doing so.

```python
import typer

my_cli_tool = typer.Typer()


@my_cli_tool.command()
def run(
    command_arg: str = typer.Option(
        ...,
        help="Compulsory arg",
        metavar="choice1/choice2",
        show_choices=True,
        prompt=False,
        case_sensitive=False,
    )
) -> None:
    """This my_cli_tool does some work"""
    # Your logic goes here
    typer.echo(f"Running my_cli_tool with arg: {command_arg}")
```

4. Now under `sima_tool/main.py`, import `my_cli_tool` as

```python
from my_new_tool.main import my_cli_tool
```

5. Add this imported tool to parent `typer` app in `sima_tool/main.py` as

```python
app.add_typer(fps_getter_app)
app.add_typer(model_to_pipeline_app)
app.add_typer(my_cli_tool) # ---> This is new entry
```

6. After this, your tool's entry will be shown under the main `sima-model-to-pipeline tool`.
