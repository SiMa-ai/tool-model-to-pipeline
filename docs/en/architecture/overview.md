# Project Overview

This tool project consists of multiple files and directories that support various functionalities.  
It follows a *__plugin-based architecture__* that allows easy addition of new steps, surgeries, or compilation logic.

| Directory / File | Description |
|------------------|-------------|
| [`main.py`](../../../model_to_pipeline/main.py) | The main entry point of the tool. |
| [`cli.py`](../../../model_to_pipeline/cli.py) | Accepts command-line arguments and returns an `argparse.Namespace` instance. |
| [`utils`](../../../model_to_pipeline/utils/) | Contains utility modules such as `loader` and process utilities for executing host commands using the `subprocess` module. Also includes `onnx_helper.py` for model graph surgery and a logging module for maintaining logs across all steps. |
| [`compilers`](../../../model_to_pipeline/compilers/) | Contains all compiler logic for different supported models. |
| [`constants`](../../../model_to_pipeline/constants/) | Stores internal constants used across the project. |
| [`pipeline`](../../../model_to_pipeline/pipeline/) | Handles pipeline creation using the compiled model’s `.tar.gz` file. It also injects plugins such as `genericboxdecode` and `overlay` with their respective config files, and generates two `application.json` files — one for `rtsp` and another for `pcie`. |
| [`surgeons`](../../../model_to_pipeline/surgeons/) | Contains all model surgeon scripts used for graph surgery before compilation. |
| [`steps`](../../../model_to_pipeline/steps/) | Defines the sequential steps executed from model surgery through final project creation, representing the complete lifecycle of the tool. |

---

## Adding New

The tool is designed with a modular, plugin-based architecture.  
To add support for new models, simply create the corresponding plugin modules — no modification of existing source files is required.

For example:
- Add a new compiler under [`compilers`](../../../model_to_pipeline/compilers/README.md)
- Add a new step under [`steps`](../../../model_to_pipeline/steps/README.md)
- Extend utilities under [`utils`](../../../model_to_pipeline/utils/README.md)

This approach ensures scalability and clean separation of functionality.
