# SiMa.ai Model-to-Pipeline Toolkit 

[![SDK Compatibility](https://img.shields.io/badge/SDK-2.1.0-blue.svg)](#) 
[![Models](https://img.shields.io/badge/Supported-YOLO-green.svg)](#) 
[![Python](https://img.shields.io/badge/Python-3.10%2B-orange.svg)](#) 
[![License](https://img.shields.io/badge/License-MIT-lightgrey.svg)](../LICENSE) 
[![한국어 문서](https://img.shields.io/badge/Docs-Korean-lightblue.svg)](README.kr.md) 

The **Model-to-Pipeline Tool** converts FP32 YOLO models into optimized GStreamer pipelines for the **SiMa.ai** platform. This guide is a step-by-step hands-on guide for customers to build and run an AI inference pipeline on SiMa.ai’s Modalix Edge AI platform using the Ultralytics YOLOv8m model as an example. By following this document, customers will experience the entire process in one go: setting up the SDK environment, converting the model, creating an mpk, and running it on Modalix. 

### 📦 Installation

See [Installation Guide](docs/en/installation.md). 

### 🚀 Usage

- [Model-to-Pipeline](docs/en/usage/model-to-pipeline.md)
- [Get-FPS](docs/en/usage/get-fps.md) 
- [Infer](docs/en/usage/infer.md)

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

### 🧠 Architecture & Design

Learn how the tool works internally: 

- [Project Overview](docs/en/architecture/overview.md) 
- [Adding New Models](docs/en/architecture/add-model.md)
- [Adding New Steps](docs/en/architecture/add-step.md)
- [Adding New Tools](docs/en/architecture/add-tool.md) 
 
### 🧾 Logs 

See [Logs Overview](logs.md)