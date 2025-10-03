# # Copyright (c) 2025 SiMa.ai
# #
# # This source code is licensed under the MIT license found in the
# # LICENSE file in the root directory of this source tree.
# #
# # Permission is hereby granted, free of charge, to any person obtaining a copy
# # of this software and associated documentation files (the "Software"), to deal
# # in the Software without restriction, including without limitation the rights
# # to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# # copies of the Software, and to permit persons to whom the Software is
# # furnished to do so, subject to the following conditions:
# #
# # The above copyright notice and this permission notice shall be included in all
# # copies or substantial portions of the Software.
# #
# # THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# # IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# # FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# # AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# # LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# # OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# # SOFTWARE.

# import argparse
# import logging
# from pathlib import Path

# from sima_model_to_pipeline.compilers.compiler_base import CompilerMeta


# def argparser():
#     parser = argparse.ArgumentParser(
#         description="Convert the model to working pipeline."
#     )

#     # Args related to the app
#     parser.add_argument(
#         "--step",
#         type=str,
#         help="Mention step name to execute. Other steps will be skipped",
#         required=False,
#         default=None,
#     )

#     # Args related to model
#     parser.add_argument(
#         "--model_path",
#         type=str,
#         help="Path to the model file to use.",
#         required=False,
#     )
#     parser.add_argument(
#         "--model_name",
#         type=str,
#         help="Name of the model.",
#         required=False,
#         choices=["yolov8", "yolov9"],
#     )
#     parser.add_argument(
#         "--post_surgery_model_path",
#         type=str,
#         help="Path to the model file after surgery.",
#         required=False,
#         default=None,
#     )
#     parser.add_argument(
#         "--model_type",
#         type=str,
#         help="Type of the model to use.",
#         choices=["object-detection", "image-classification", "segmentation"],
#         default="object-detection",
#     )
#     parser.add_argument(
#         "--input_width",
#         type=int,
#         help="Input width of the pipeline",
#         required=False,
#         default=None,
#     )
#     parser.add_argument(
#         "--input_height",
#         type=int,
#         help="Input height of the pipeline",
#         required=False,
#         default=None,
#     )

#     # Args related to compilation step
#     parser.add_argument(
#         "--compilation_result_dir",
#         type=str,
#         help="Model compilation result dir. Compiled model will be dumped here",
#         required=False,
#         default="result",
#     )
#     parser.add_argument(
#         "--compiler",
#         type=str,
#         help="Name of compiler to choose from. This will raise exception if provided compiler is not found. If not provided, it'll pick the one registered with model name",
#         required=False,
#         default=list(CompilerMeta.registry.keys())[0],
#         choices=list(CompilerMeta.registry.keys()),
#     )
#     parser.add_argument(
#         "--calibration_data_path",
#         type=str,
#         help="Path to the calibration dataset. If not provided, the model will be used as is.",
#         required=False,
#     )
#     parser.add_argument(
#         "--calibration_samples_count",
#         type=int,
#         help="Max number of calibration samples. If not given, keeping to maximum images available in dataset",
#         default=None,
#     )
#     parser.add_argument(
#         "--arm_only",
#         action="store_true",
#         help="If set, the model will be compiled for ARM architecture only.",
#         default=False,
#     )
#     parser.add_argument(
#         "--asym",
#         action="store_true",
#         help="If set, the model will be compiled with asym quantization",
#         default=False,
#     )
#     parser.add_argument(
#         "--per_ch",
#         action="store_true",
#         help="If set, the model will be compiled with per_ch quantization",
#         default=False,
#     )
#     parser.add_argument(
#         "--bias_correction",
#         action="store_true",
#         help="If set, the model will be compiled with bias_correction",
#         default=False,
#     )
#     parser.add_argument(
#         "--bf16", action="store_true", help="Keep precision as bf16?", default=False
#     )
#     parser.add_argument(
#         "--ceq", action="store_true", help="Enable channel equalization", default=False
#     )
#     parser.add_argument(
#         "--compress",
#         action="store_true",
#         help="Compress while compiling",
#         default=False,
#     )
#     parser.add_argument(
#         "--nbits",
#         help="INT Bit precision for compilation",
#         default=8,
#         type=int,
#         choices=[8, 16],
#     )
#     parser.add_argument(
#         "--mode",
#         help="Requantization Mode",
#         default="sima",
#         type=str,
#         choices=["sima", "tflite"],
#     )
#     parser.add_argument(
#         "--calibration_type",
#         type=str,
#         help="Type of the Calibration to use",
#         choices=["min_max", "moving_average", "entropy", "percentile", "mse"],
#         required=False,
#         default="mse",
#     )
#     parser.add_argument("--batch_size", help="Batch size", default=1, type=int)
#     parser.add_argument(
#         "--calibration_ds_extn",
#         default="jpg",
#         help="Extension of the calibration dataset files. Default is .jpg.",
#         type=str,
#     )
#     parser.add_argument(
#         "--device_type",
#         type=str,
#         help="Type of the board to use for compilation.",
#         choices=["davinci", "modalix", "both"],
#         required=False,
#         default="davinci",
#     )

#     # Args related to pipeline creation
#     parser.add_argument(
#         "--input_resource",
#         type=str,
#         help="Path to input image or video.",
#         required=False,
#         default=None,
#     )
#     parser.add_argument(
#         "--pipeline_name",
#         type=str,
#         help="Final name of the pipeline.",
#         required=False,
#         default=None,
#     )
#     parser.add_argument(
#         "--config_yaml",
#         "-c",
#         help="Provide configuration from config yaml file",
#         required=False,
#         default=None,
#     )

#     # Extra params
#     parser.add_argument(
#         "--no_box_decode",
#         action="store_true",
#         help="Provide this arg if generic simaaiboxdecode is not required. Instead, detessdequant will be used.",
#         required=False,
#         default=False,
#     )
#     parser.add_argument(
#         "--rtsp_src",
#         type=str,
#         help="RTSP Stream in case of RTSP pipeline",
#         required=False,
#         default=None,
#     )
#     parser.add_argument(
#         "--host_ip",
#         type=str,
#         help="Host IP to stream back the video in case of RTSP pipeline",
#         required=False,
#         default=None,
#     )
#     parser.add_argument(
#         "--host_port",
#         type=str,
#         help="Host port to stream back the video in case of RTSP pipeline",
#         required=False,
#         default=None,
#     )
#     parser.add_argument(
#         "--detection_threshold",
#         type=float,
#         help="Input detection_threshold of the pipeline",
#         required=False,
#         default=0.4,
#     )
#     parser.add_argument(
#         "--nms_iou_threshold",
#         type=float,
#         help="Input nms_iou_threshold of the pipeline",
#         required=False,
#         default=0.4,
#     )
#     parser.add_argument(
#         "--topk",
#         type=int,
#         help="Expected topk value for the pipeline",
#         required=False,
#         default=25,
#     )
#     parser.add_argument(
#         "--labels_file",
#         type=str,
#         help="Path to the labels file for the model.",
#         required=False,
#         default=None,
#     )

#     args = parser.parse_args()

#     if not args.config_yaml:
#         missing_args = []
#         if not args.model_path:
#             missing_args.append("--model_path")
#         if not args.model_name:
#             missing_args.append("--model_name")
#         if not args.pipeline_name:
#             missing_args.append("--pipeline_name")

#         if missing_args:
#             parser.error(message=f"Missing required args: {' '.join(missing_args)}")

#     else:
#         logging.info(
#             f"Overriding the params from commandline with parameters from {args.config_yaml}"
#         )
#         assert Path(args.config_yaml).exists(), f"{args.config_yaml} doesn't exist"
#         import yaml

#         with open(args.config_yaml, "r") as config:
#             config_data = yaml.safe_load(config)

#         args.pipeline_name = config_data.get("pipeline_name", "MyPipeline")
#         # Overriding model params
#         args.model_path = config_data.get("model_params", {}).get(
#             "model_path", args.model_path
#         )
#         args.model_name = config_data.get("model_params", {}).get(
#             "model_name", args.model_name
#         )
#         args.post_surgery_model_path = config_data.get("model_params", {}).get(
#             "post_surgery_model_path", args.post_surgery_model_path
#         )
#         args.model_type = config_data.get("model_params", {}).get(
#             "model_type", args.model_type
#         )
#         args.input_width = config_data.get("model_params", {}).get(
#             "input_width", args.input_width
#         )
#         args.input_height = config_data.get("model_params", {}).get(
#             "input_height", args.input_height
#         )

#         # Overriding compilation params
#         args.compilation_result_dir = config_data.get("compilation_params", {}).get(
#             "compilation_result_dir", args.compilation_result_dir
#         )
#         args.compiler = config_data.get("compilation_params", {}).get(
#             "compiler", args.compiler
#         )
#         args.batch_size = config_data.get("compilation_params", {}).get(
#             "batch_size", args.batch_size
#         )
#         args.batch_size = config_data.get("compilation_params", {}).get(
#             "batch_size", args.batch_size
#         )
#         args.device_type = config_data.get("compilation_params", {}).get(
#             "device_type", args.device_type
#         )
#         # Calibration params
#         args.calibration_data_path = (
#             config_data.get("compilation_params", {})
#             .get("calibration_params", {})
#             .get("calibration_data_path", args.calibration_data_path)
#         )
#         args.calibration_ds_extn = (
#             config_data.get("compilation_params", {})
#             .get("calibration_params", {})
#             .get("calibration_ds_extn", args.calibration_ds_extn)
#         )
#         args.calibration_samples_count = (
#             config_data.get("compilation_params", {})
#             .get("calibration_params", {})
#             .get("calibration_samples_count", args.calibration_samples_count)
#         )
#         args.calibration_type = (
#             config_data.get("compilation_params", {})
#             .get("calibration_params", {})
#             .get("calibration_type", args.calibration_type)
#         )
#         # Quantization config
#         args.asym = (
#             config_data.get("compilation_params", {})
#             .get("quant_config", {})
#             .get("asym", args.asym)
#         )
#         args.per_ch = (
#             config_data.get("compilation_params", {})
#             .get("quant_config", {})
#             .get("per_ch", args.per_ch)
#         )
#         args.bias_correction = (
#             config_data.get("compilation_params", {})
#             .get("quant_config", {})
#             .get("bias_correction", args.bias_correction)
#         )
#         args.bf16 = (
#             config_data.get("compilation_params", {})
#             .get("quant_config", {})
#             .get("bf16", args.bf16)
#         )
#         args.ceq = (
#             config_data.get("compilation_params", {})
#             .get("quant_config", {})
#             .get("ceq", args.ceq)
#         )
#         args.compress = (
#             config_data.get("compilation_params", {})
#             .get("quant_config", {})
#             .get("compress", args.compress)
#         )
#         args.nbits = (
#             config_data.get("compilation_params", {})
#             .get("quant_config", {})
#             .get("nbits", args.nbits)
#         )
#         args.mode = (
#             config_data.get("compilation_params", {})
#             .get("quant_config", {})
#             .get("mode", args.mode)
#         )

#         # Extra params
#         args.no_box_decode = config_data.get("extras", {}).get(
#             "no_box_decode", args.mode
#         )
#         args.rtsp_src = config_data.get("extras", {}).get("rtsp_src", args.mode)
#         args.host_ip = config_data.get("extras", {}).get("host_ip", args.mode)
#         args.host_port = config_data.get("extras", {}).get("host_port", args.mode)

#     return args
