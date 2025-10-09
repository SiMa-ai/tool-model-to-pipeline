#!/usr/bin/env python3
"""
Step 9: Compile post-surgery ONNX → .tar.gz artifact

Goal:
- Write the artifact to exactly: ./result/modalix/yolov8m_pipeline.tar.gz
  (i.e., directory "./result/modalix" with a trailing dot; filename "yolov8m_pipeline.tar.gz")

Usage examples:

  # Specify calibration folder and limit number of calibration images
  python3 exam_compile.py \
    --surgery-onnx ./yolov8m_pipeline_mod.onnx \
    --precision int8 \
    --calib-dir /home/docker/calibration_images \
    --calib-count 128 \
    --calib-ext jpg \
    --calib-type minmax \
    --batch-size 1 \
    --device-type modalix

All parameters:

  --surgery-onnx    Path to the post-surgery ONNX (required)
  --precision       Quantization mode: int8 | bf16 | fp16  [default: int8]
  --calib-dir       Folder with calibration images          [default: /home/docker/calibration_images]
  --calib-count     Number of calibration images to use     [default: (tool default)]
  --calib-ext       Calibration image extension (jpg/png)   [default: jpg]
  --calib-type      Calibration method (e.g., minmax)       [default: minmax]
  --batch-size      Batch size for compile                  [default: 1]
  --device-type     modalix | davinci | both                [default: modalix]
  --arm-only        Use ARM-only flows (flag)
  --act-asym        Activation asymmetric quant (flag)
  --act-per-ch      Activation per-channel quant (flag)
  --act-nbits       Activation bit width                    [default: 8]
  --wt-asym         Weight asymmetric quant (flag)
  --wt-per-ch       Weight per-channel quant (flag)
  --wt-nbits        Weight bit width                        [default: 8]
  --bias-correction Enable bias correction (flag)
  --ceq             Enable Cross-layer Equalization (flag)  [default: True]
  --smooth-quant    Enable SmoothQuant (flag)
  --compress        Compress artifacts (flag)               [default: True]
  --compiler        Override compiler key (advanced)
  --model-name      Model family name                       [default: yolov8]
  --pipeline-name   Pipeline logical name                   [default: yolov8m_pipeline]

Notes:
- Output directory is forced to ./result/modalix. (with trailing dot).
- Output filename is forced to yolov8m_pipeline.tar.gz
- You can still change compile parameters, but not the output location/name (as requested).
"""

import argparse, sys
from pathlib import Path
from model_to_pipeline.compilers.compile_yolo_generic import CompileYoloGeneric


# Fixed output location and name (per requirement)
FIXED_OUT_DIR = Path("./result").resolve()   # directory with trailing dot
FIXED_TAR_NAME = "yolov8m_pipeline.tar.gz"           # exact filename


def build_args(ns):
    class A: pass
    a = A()

    # Model/pipeline identity
    a.model_name = ns.model_name                     # default "yolov8"
    a.pipeline_name = ns.pipeline_name               # default "yolov8m_pipeline"

    # Inputs
    a.post_surgery_model_path = str(Path(ns.surgery_onnx).resolve())

    # Force output directory to the requested path and ensure it exists
    FIXED_OUT_DIR.mkdir(parents=True, exist_ok=True)
    a.compilation_result_dir = str(FIXED_OUT_DIR)

    # Compiler override (optional/advanced)
    a.compiler = getattr(ns, "compiler", None)

    # Quantization / precision
    a.mode = ns.precision                            # int8/bf16/fp16
    a.batch_size = ns.batch_size

    # Calibration
    a.calibration_data_path = ns.calib_dir
    a.calibration_samples_count = ns.calib_count
    a.calibration_ds_extn = getattr(ns, "calib_ext", "jpg")
    a.calibration_type = getattr(ns, "calib_type", "minmax")

    # Device toggles
    a.device_type = ns.device_type
    a.arm_only = ns.arm_only

    # Fine-tuning quant flags (tool may ignore some)
    a.act_asym = ns.act_asym
    a.act_per_ch = ns.act_per_ch
    a.act_bf16 = (ns.precision == "bf16")
    a.act_nbits = ns.act_nbits
    a.wt_asym = ns.wt_asym
    a.wt_per_ch = ns.wt_per_ch
    a.wt_bf16 = (ns.precision == "bf16")
    a.wt_nbits = ns.wt_nbits
    a.bias_correction = ns.bias_correction
    a.ceq = ns.ceq
    a.smooth_quant = ns.smooth_quant
    a.compress = ns.compress

    return a


def expected_artifact_path() -> Path:
    """Return the exact requested artifact path."""
    return FIXED_OUT_DIR / FIXED_TAR_NAME


def main():
    ap = argparse.ArgumentParser(description="Compile YOLOv8m post-surgery ONNX to fixed artifact path.")
    ap.add_argument("--surgery-onnx", required=True, help="Path to post-surgery ONNX (e.g., yolov8m_pipeline_mod.onnx)")
    ap.add_argument("--precision", default="int8", choices=["int8", "bf16", "fp16"])
    ap.add_argument("--calib-dir", default="/home/docker/calibration_images")
    ap.add_argument("--calib-count", type=int, default=None)
    ap.add_argument("--calib-ext", default="jpg")
    ap.add_argument("--calib-type", default="minmax")
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--device-type", default="modalix", choices=["modalix", "davinci", "both"])
    ap.add_argument("--arm-only", action="store_true")

    # Fine-tuning quant flags
    ap.add_argument("--act-asym", action="store_true")
    ap.add_argument("--act-per-ch", action="store_true")
    ap.add_argument("--act-nbits", type=int, default=8)
    ap.add_argument("--wt-asym", action="store_true")
    ap.add_argument("--wt-per-ch", action="store_true")
    ap.add_argument("--wt-nbits", type=int, default=8)
    ap.add_argument("--bias-correction", action="store_true")
    ap.add_argument("--ceq", action="store_true", default=True)
    ap.add_argument("--smooth-quant", action="store_true")
    ap.add_argument("--compress", action="store_true", default=True)

    # Advanced / identity
    ap.add_argument("--compiler", default=None)
    ap.add_argument("--model-name", default="yolov8")
    ap.add_argument("--pipeline-name", default="yolov8m_pipeline")

    args = ap.parse_args()
    args_ns = build_args(args)

    ok = CompileYoloGeneric().run(args_ns)
    if not ok:
        sys.exit(1)

    # Some toolchains place the .tar.gz inside the compilation_result_dir with a name derived
    # from pipeline_name. We don't rename files here; we just tell the user the target path.
    target = expected_artifact_path()
    print("[OK] Compilation finished.")
    print("Requested artifact path:", target)
    print("If your tool emitted a different name, move/rename it to match:")
    print(f"  mv \"{FIXED_OUT_DIR}/{args_ns.pipeline_name}.tar.gz\" \"{target}\"  # adjust if needed")


if __name__ == "__main__":
    main()
