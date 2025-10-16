#!/usr/bin/env python3
"""
Step 8: Graph Surgery for YOLOv8m → produces yolov8m_pipeline_mod.onnx
(Output is always written to the current working directory; no subfolders.)

Goal:
- Perform YOLOv8m graph surgery on an input ONNX so it is ready for SiMa.ai pipelines.
- Normalize I/O (width/height), bake class count and Top-K, optionally attach label mapping.
- Keep YOLO detection bbox decode ENABLED by default (SDK handles decode).
- Always write the output to the current working directory as: ./yolov8m_pipeline_mod.onnx

Usage examples:
    python3 exam_surgery.py \
        --model-path ./yolov8m.onnx \
        --model-name yolov8 \
        --pipeline-name yolov8m_pipeline \
        --num-classes 80 \
        --input-width 1920 \
        --input-height 1080 \
        --labels-file "" \
        --topk 300 \
        --device-type modalix

All parameters:

  --model-path <path>
      Path to the input YOLOv8m ONNX file. (Required)

  --model-name <str> [default: yolov8]
      Logical model family name. Usually kept as "yolov8".

  --pipeline-name <str> [default: yolov8m_pipeline]
      Logical pipeline name to be referenced by later steps.

  --num-classes <int> [default: 80]
      Number of detection classes to include in the modified ONNX.

  --input-width <int> [default: 1920]
      Input tensor width (pixels). Used to reshape and normalize the model input.

  --input-height <int> [default: 1080]
      Input tensor height (pixels). Used to reshape and normalize the model input.

  --labels-file <path or ""> [default: None]
      Optional label file (one class name per line).
      If omitted or passed as an empty string, name mapping is skipped.

  --topk <int> [default: 300]
      Maximum number of detections to keep in postprocessing.
      Used as a hint for the SDK pipeline.

  --no_box_decode [flag]
      Outputs raw prediction tensors without bounding-box decode.
      *Not recommended for YOLO detection models — leave this flag unset.*

  --device-type {modalix|davinci|both} [default: modalix]
      Target device type for optimization hints.

Output:
  - Always writes to: ./yolov8m_pipeline_mod.onnx
  - Output is placed directly in the current working directory (no subfolders).

Notes:
- YOLOv8m (detection) use-case: do NOT use --no_box_decode (SDK handles decode).
- labels-file is optional; omit or pass empty string if you don't need name mapping.
"""
import argparse
import sys
from pathlib import Path

# Import YOLOv8 surgeon
import model_to_pipeline.surgeons.surgeon_yolov8  # noqa: F401
from model_to_pipeline.steps.step_surgery import StepSurgery


def build_args(ns):
    class A: ...
    a = A()
    # Inputs
    a.model_path = str(Path(ns.model_path).resolve())
    a.model_name = ns.model_name                # "yolov8"
    a.num_classes = ns.num_classes
    a.input_width = ns.input_width
    a.input_height = ns.input_height
    # Output fixed to current working directory
    a.pipeline_name = ns.pipeline_name          # "yolov8m_pipeline"
    a.post_surgery_model_path = str(Path(".").resolve() / "yolov8m_pipeline_mod.onnx")
    # Optional defaults
    a.labels_file = ns.labels_file if ns.labels_file else None
    a.topk = ns.topk
    a.no_box_decode = ns.no_box_decode          # For detection: leave False (do not set)
    a.config_yaml = None
    a.device_type = ns.device_type
    return a


def main():
    ap = argparse.ArgumentParser(description="Graph surgery for YOLOv8m → yolov8m_pipeline_mod.onnx")
    ap.add_argument("--model-path", required=True, help="Path to input YOLOv8m ONNX")
    ap.add_argument("--model-name", default="yolov8", help="Model family name (e.g., yolov8)")
    ap.add_argument("--pipeline-name", default="yolov8m_pipeline", help="Logical pipeline name")
    ap.add_argument("--num-classes", type=int, default=80, help="Number of classes (default: 80)")
    ap.add_argument("--input-width", type=int, default=1920, help="Model input width")
    ap.add_argument("--input-height", type=int, default=1080, help="Model input height")
    ap.add_argument("--labels-file", default=None, help="Optional labels file (txt). Omit/empty if not needed.")
    ap.add_argument("--topk", type=int, default=300, help="Top-N detections to keep (postprocess hint)")
    ap.add_argument("--no_box_decode", action="store_true",
                    help="Output raw tensors without bbox decode. (Not recommended for YOLO detection.)")
    ap.add_argument("--device-type", default="modalix", choices=["modalix", "davinci", "both"],
                    help="Target device type")
    args = ap.parse_args()

    args_ns = build_args(args)
    ok = StepSurgery().run(args_ns)
    if not ok:
        sys.exit(1)
    print("[OK] Surgery ONNX created:", args_ns.post_surgery_model_path)


if __name__ == "__main__":
    main()
