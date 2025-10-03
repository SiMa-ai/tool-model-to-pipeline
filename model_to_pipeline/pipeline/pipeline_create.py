# Copyright (c) 2025 SiMa.ai
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.


import argparse
from glob import glob
import json
import logging
import os
from pathlib import Path
import re
import shutil
from tempfile import TemporaryDirectory
from typing import Tuple, Union

from model_to_pipeline.constants.yolo_labels import YOLO_80_LABELS
from model_to_pipeline.pipeline.pipeline_base import PipelineBase
from model_to_pipeline.utils.process_util import execute_command
import copy


class PipelineCreate(PipelineBase):
    """Pipeline for creating a new project with the compiled model.

    This pipeline is responsible for creating a new project directory, copying the compiled model,
    and setting up the necessary files for the project.
    """

    name = "create_pipeline"
    sequence = 1

    def run(self, args: argparse.Namespace) -> bool:
        """Run the pipeline to create a new project.

        Args:
            args (argparse.Namespace): Command line arguments.

        Returns:
            bool: True if the pipeline ran successfully, False otherwise.
        """
        input_resource = (
            args.input_resource
            if args.input_resource
            else glob(args.calibration_data_path + "/*." + args.calibration_ds_extn)[0]
        )
        davinci_model, modalix_model = self.get_compiled_models(args=args)
        # See if davinci path is availble, else use modalix. If both availble, use davinci
        model_path = davinci_model or modalix_model
        logging.info(f"Creating a pipeline with {model_path}")
        command = f"mpk project create --model-path {model_path} --input-resource {input_resource}"
        logging.info(f"Executing command to create pipeline: {command}")
        op, ec = execute_command(command=command)

        pipeline_name = model_path.split("/")[-1].split(".")[0] 

        pipeline_dir = pipeline_name + '_simaaisrc'
        print('created pipeline_dir:', pipeline_dir)

        # if os.path.exists(pipeline_name):
        #     print('pipeline_dir:', pipeline_name,' folder exists, removing it')
        #     shutil.rmtree(pipeline_name)
        # logging.info(f"Renaming pipeline directory from {pipeline_dir} to {pipeline_name}")
        
        # os.rename(pipeline_dir, pipeline_name)
        args.pipeline_name = pipeline_dir

        list(map(logging.info, op.splitlines()))
        assert ec == 0, "Failed to create pipeline project."
        self.prepare_for_multimodel(
            args=args, model_paths=[davinci_model, modalix_model]
        )
        self.prepare_plugins(args)
        self.update_gst_commands(args)
        logging.info("Pipeline creation completed.")
        return True

    def prepare_for_multimodel(
        self, args: argparse.Namespace, model_paths: Tuple[str]
    ) -> None:
        plugins_to_prepare = ["processcvu", "processmla"]
        plugin_base_dir = Path(args.pipeline_name) / "plugins"
        for model in model_paths:
            if not model:
                # Returning if model files are not found
                continue
            logging.info(f"Working for model: {model}")
            regex = re.compile(f".*{args.compilation_result_dir}/(?P<device>\w+)/.*")
            matching = regex.match(model)
            if matching:
                device = matching.group("device")
            with TemporaryDirectory() as tmpdir:
                shutil.unpack_archive(filename=model, extract_dir=tmpdir)
                # Finding all postproc, preproc, process_mla, model_elf
                postproc, preproc, process_mla, model_elf = [
                    file
                    for file in sorted(os.listdir(tmpdir))
                    if file.endswith("elf")
                    or file.endswith("mla.json")
                    or file.endswith("postproc.json")
                    or file.endswith("preproc.json")
                ]
                mpk_json = [
                    file for file in os.listdir(tmpdir) if file.endswith("mpk.json")
                ]
                mpk_json = mpk_json[0] if mpk_json else None

                if mpk_json:
                    (Path(args.pipeline_name) / "resources").mkdir(exist_ok=True)
                    shutil.copy(
                        src=Path(tmpdir) / mpk_json,
                        dst=Path(args.pipeline_name) / "resources"/ "mpk.json"
                    )
                    application_json_path = Path(args.pipeline_name) / "application.json"
                    with open(application_json_path, "r") as app_json_file:
                        application_json = json.load(app_json_file)
                        print(f'application_json pipelines-0-resources: {application_json["pipelines"][0]["resources"]}')
                        application_json["pipelines"][0]["resources"] = [{
                                                                            "name": "mpk.json",
                                                                            "location": "resources/mpk.json",
                                                                            "destination": {
                                                                                "provider": "RPM",
                                                                                "type": "config"
                                                                            }
                                                                        }]

                    with open(application_json_path, "w+") as file:
                        json.dump(application_json, file, indent=4)
                    

                logging.info(
                    f"mpk json file have been added to resource folder for cpp/python wrapper usage. - {mpk_json}"
                )

                logging.info(
                    f"Found files from compiled model as : {postproc, preproc, process_mla, model_elf}"
                )

                # Prepare
                for plugin in plugins_to_prepare:
                    plugin_dir = Path(plugin_base_dir) / plugin
                    logging.info(f"Making directory {plugin_dir / device}")
                    (plugin_dir / device).mkdir(exist_ok=True)
                    (plugin_dir / device).mkdir(exist_ok=True)
                    logging.info(
                        f'Making directory {plugin_dir / device / "cfg"} and {plugin_dir / device / "res"} under {device} directory'
                    )
                    (plugin_dir / device / "cfg").mkdir(exist_ok=True)
                    (plugin_dir / device / "res").mkdir(exist_ok=True)

                    if plugin == "processcvu":
                        shutil.copy(
                            src=Path(tmpdir) / preproc, dst=plugin_dir / device / "cfg"
                        )
                        shutil.copy(
                            src=Path(tmpdir) / postproc, dst=plugin_dir / device / "cfg"
                        )
                    # if plugin == "detesdequant":
                    #     shutil.copy(
                    #         src=Path(tmpdir) / postproc, dst=plugin_dir / device / "cfg"
                    #     )
                    if plugin == "processmla":
                        shutil.copy(
                            src=Path(tmpdir) / process_mla,
                            dst=plugin_dir / device / "cfg",
                        )
                        shutil.copy(
                            src=Path(tmpdir) / model_elf,
                            dst=plugin_dir / device / "res",
                        )

                    logging.info(
                        f"Removing older cfg and res folder from plugin {plugin}"
                    )
                    shutil.rmtree(plugin_base_dir / plugin / "cfg", ignore_errors=True)
                    shutil.rmtree(plugin_base_dir / plugin / "res", ignore_errors=True)

    def get_compiled_models(self, args: argparse.Namespace) -> Tuple[str, str]:
        """Gets compiled models from result directory. Helful when creating pipeline for both the platforms

        Args:
            args (argparse.Namespace): Commandline args

        Returns:
            str: Tuple of model paths
        """
        mod_model_name = (
            args.post_surgery_model_path.split(".")[0]
            if args.post_surgery_model_path
            else args.pipeline_name
        )
        davinci_model = glob(args.compilation_result_dir + f"/davinci/*.tar.gz")
        modalix_model = glob(args.compilation_result_dir + f"/modalix/*.tar.gz")
        # If the model is found, pick the 0th from the obtained list
        if davinci_model:
            davinci_model = args.compilation_result_dir + f"/davinci/{args.pipeline_name}.tar.gz" # davinci_model[0]
            logging.info(f"Found davinci model: {davinci_model}")
        if modalix_model:
            modalix_model = args.compilation_result_dir + f"/modalix/{args.pipeline_name}.tar.gz" # modalix_model[0]
            logging.info(f"Found modalix model: {modalix_model}")
        if args.device_type == "davinci":
            # Removing modalix model from consideration
            modalix_model = []
        if args.device_type == "modalix":
            # Removing davinci model from consideration
            davinci_model = []

        return davinci_model, modalix_model

    def update_plugin_entry(
        self, plugin_name: str, args: argparse.Namespace, resources: Union[None, dict]
    ) -> None:
        """Update the plugin entry in the project directory.

        Args:
            plugin_name (str): Name of the plugin to update.
        """
        application_json_path = Path(args.pipeline_name) / "application.json"
        plugins_info_path = Path(args.pipeline_name) / ".project" / "pluginsInfo.json"
        logging.info(f"Updating plugin entry for: {plugin_name}")

        application_json_entry = {
            "name": plugin_name,
            "pluginName": plugin_name,
            "gstreamerPluginName": plugin_name,
            "pluginGid": plugin_name,
            "pluginId": f"{plugin_name}_1",
            "sequence": 1,
            "resources": resources,
        }
        plugins_info_entry = {
            "gid": plugin_name,
            "path": f"plugins/{plugin_name}",
        }

        # Update application.json
        with open(application_json_path, "r") as file:
            application_json = json.load(file)

        application_json["pipelines"][0]["plugins"].append(application_json_entry)
        logging.info(
            f'Plugins info in the application.json: {application_json["pipelines"][0]["plugins"]}'
        )
        logging.info(f'device type: {args.device_type}')
        # Replace "cfg/" with "davinci/cfg/" in configs paths of all plugins
        for plugin in application_json["pipelines"][0]["plugins"]:
            if plugin["pluginGid"] == "processcvu":
                if "resources" in plugin and "configs" in plugin["resources"]:
                    plugin["resources"]["configs"] = [
                        config.replace("cfg/", f"{args.device_type}/cfg/") if isinstance(config, str) and f"{args.device_type}/cfg/" not in config else config
                        for config in plugin["resources"]["configs"]
                    ]

            if plugin["pluginGid"] == "processmla":
                if "resources" in plugin and "configs" in plugin["resources"]:
                    plugin["resources"]["configs"] = [
                        config.replace("cfg/", f"{args.device_type}/cfg/") if isinstance(config, str) and f"{args.device_type}/cfg/" not in config else config
                        for config in plugin["resources"]["configs"]
                    ]
                    plugin["resources"]["shared"] = [
                        config.replace("res/", f"{args.device_type}/res/") if isinstance(config, str) and f"{args.device_type}/res/" not in config else config
                        for config in plugin["resources"]["shared"]
                    ]


        with open(application_json_path, "w") as file:
            json.dump(application_json, file, indent=4)

        # Update pluginsInfo.json
        with open(plugins_info_path, "r") as file:
            plugins_info = json.load(file)
        plugins_info["pluginsInfo"].append(plugins_info_entry)
        with open(plugins_info_path, "w") as file:
            json.dump(plugins_info, file, indent=4)

    @staticmethod
    def calculate_buffer_for_pcie_sink(inp_depth, inp_width, inp_height):
        """Calculate buffer size for the pciesocsink plugin for the models
          not using box decoder and sending the postproc buffer as a whole 
          to the sink to process the postprocessing steps on the host.
        Args:
            args : inp_depth - This is a array of model output nodes depth
            args : inp_width - This is a array of model output nodes width
            args : inp_height - This is a array of model output nodes height
        Returns:
            int: Calculated buffer size of detess/dequant
        """
            
        return sum([int(depth)*int(width)*int(height) for (depth, width, height) in zip(inp_depth, inp_width, inp_height)])*4

    def prepare_plugins(self, args: argparse.Namespace) -> None:
        """Prepare necessary plugins in the project directory.

        Args:
            args (argparse.Namespace): Command line arguments.
        """
        with TemporaryDirectory() as temp_dir:
            # tar_file_path = f"{args.pipeline_name}.tar.gz"
            tar_file_path = glob(
                args.compilation_result_dir + "/**/*.tar.gz", recursive=True
            )[0]
            shutil.unpack_archive(tar_file_path, temp_dir)
            plugins_dir = Path(args.pipeline_name) / "plugins"
            plugin_zoo_path = Path(
                "/usr/local/simaai/plugin_zoo/gst-simaai-plugins-base/gst/"
            )
            plugins_to_copy = ["genericboxdecode", "overlay", "templates"]
            for plugin in plugins_to_copy:
                logging.info(f"Preparing plugin: {plugin}")
                plugin_path = plugin_zoo_path / plugin
                shutil.copytree
                shutil.copytree(
                    src=plugin_path,
                    dst=plugins_dir / plugin,
                    dirs_exist_ok=True,
                )
                match plugin:
                    case "genericboxdecode":
                        Path(plugin_path / "cfg").mkdir(parents=False, exist_ok=True)
                        if not (
                            plugins_dir / plugin / "cfg" / "boxdecoder.json"
                        ).exists():
                            os.makedirs(plugins_dir / plugin / "cfg", exist_ok=True)
                            logging.info(
                                "Copying boxdecoder.json from temp directory to plugin directory"
                            )
                            shutil.move(
                                src=Path(temp_dir) / "0_boxdecoder.json",
                                dst=plugins_dir / plugin / "cfg" / "boxdecoder.json",
                            )
                        self.update_plugin_entry(
                            plugin_name=plugin,
                            args=args,
                            resources={
                                "configs": [
                                    "cfg/boxdecoder.json",
                                ]
                            },
                        )

                    case "overlay":
                        Path(plugins_dir / plugin / "res").mkdir(
                            parents=False, exist_ok=True
                        )

                        if args.labels_file:
                            shutil.copy(
                                src=args.labels_file,
                                dst=plugins_dir / plugin / "res" / "labels",
                            )

                        else:
                            # If no labels file is provided, use the default YOLO 80 labels
                            logging.info("Using default YOLO 80 labels.")
                            with open(
                                plugins_dir / plugin / "res" / "labels", "w"
                            ) as f:
                                f.write("\n".join(YOLO_80_LABELS))

                        self.update_plugin_entry(
                            plugin_name=plugin,
                            args=args,
                            resources={
                                "binaries": [
                                    "res/labels",
                                ]
                            },
                        )

            # Modifying the mla.json to correct caps.
            if args.device_type == "both":
                device_types = ["davinci", "modalix"]
            else:
                device_types = [args.device_type]

            for device_type in device_types:
                logging.info("Updating process_mla.json")
                with open(Path(temp_dir) / "0_process_mla.json") as mla_json:
                    data = json.load(mla_json)
                data["input_width"] = data["output_width"]
                data["input_height"] = data["output_height"]
                data["input_depth"] = data["output_depth"]
                data.pop("output_width")
                data.pop("output_height")
                data.pop("output_depth")

                # Reading sinkpad from boxdecoder.json to match with mla.json
                with open(
                    Path(plugins_dir) / "genericboxdecode" / "cfg" / "boxdecoder.json"
                ) as boxdecoder:
                    # Copying sink pads info from boxdecoder.json to mla.json's src pads
                    data["caps"]["src_pads"] = json.load(boxdecoder)["caps"][
                        "sink_pads"
                    ]
                    # Updating the path of elf file
                    model_file = data["simaai__params"]["model_path"]
                    if not model_file.startswith("/"):
                        data["simaai__params"][
                            "model_path"
                        ] = f"/data/simaai/applications/{args.pipeline_name}/share/processmla/{model_file}"

                # Updating actual process_mla.json
                with open(
                    Path(plugins_dir)
                    / "processmla"
                    / device_type
                    / "cfg"
                    / "0_process_mla.json",
                    "w",
                ) as mla_json:
                    json.dump(obj=data, fp=mla_json, indent=4)

                # Updating preproc.json
                logging.info("Updating preproc.json")
                with open(
                    Path(plugins_dir)
                    / "processcvu"
                    / device_type
                    / "cfg"
                    / "0_preproc.json",
                    "r",
                ) as preproc_json:
                    preproc_config = json.load(preproc_json)

                # If input height and width is provided, use it else update with model's shape
                new_input_height = (
                    args.input_height
                    if args.input_height
                    else preproc_config["output_height"]
                )
                new_input_width = (
                    args.input_width
                    if args.input_width
                    else preproc_config["output_width"]
                )

                model_input_height = preproc_config["output_height"]
                model_input_width = preproc_config["output_width"]

                preproc_config["input_width"] = new_input_width
                preproc_config["input_height"] = new_input_height
                preproc_config["aspect_ratio"] = True

                if args.rtsp_src:
                    preproc_config["input_buffers"][0]["name"] = "decoder"
                else:
                    preproc_config["input_buffers"][0]["name"] = "simaaipciesrc"
                    preproc_config["input_img_type"] = (
                        "RGB"  # Setting default input image type to RGB
                    )
                    preproc_config["caps"]["sink_pads"][0][
                        "params"
                    ] = []  # Reset sink pads to accept PCIe input

                # Writing it back to json file
                with open(
                    Path(plugins_dir)
                    / "processcvu"
                    / device_type
                    / "cfg"
                    / "0_preproc.json",
                    "w",
                ) as preproc_json:
                    json.dump(obj=preproc_config, fp=preproc_json, indent=4)

                # Updating boxdecoder.json to match input height and width
                with open(
                    Path(plugins_dir) / "genericboxdecode" / "cfg" / "boxdecoder.json"
                ) as boxdecoder:
                    boxdecoder_data = json.load(boxdecoder)
                boxdecoder_data["original_height"] = new_input_height
                boxdecoder_data["original_width"] = new_input_width

                logging.info(
                    "Updating boxdecoder.json with input dimensions and detection thresholds"
                )
                logging.info(f"args.detection_threshold: {args.detection_threshold}")
                logging.info(f"args.nms_iou_threshold: {args.nms_iou_threshold}")

                boxdecoder_data["detection_threshold"] = args.detection_threshold
                boxdecoder_data["nms_iou_threshold"] = args.nms_iou_threshold
                boxdecoder_data["topk"] = args.topk
                boxdecoder_data["buffers"]["output"]["size"] = int(4 + 24 * args.topk)
                boxdecoder_data["num_classes"] = args.num_classes

                # If labels file is provided, read it and update boxdecoder.json
                if args.labels_file:
                    logging.info(f"Reading labels from {args.labels_file}")
                    with open(args.labels_file, "r") as f:
                        labels = [line.strip() for line in f.readlines()]
                        labels = [label for label in labels if label.strip()]
                    boxdecoder_data["labels"] = labels
                else:
                    # If no labels file is provided, use the default YOLO 80 labels
                    logging.info("Using default YOLO 80 labels.")
                    if args.num_classes == 80:
                        boxdecoder_data["labels"] = YOLO_80_LABELS
                    else:
                        logging.warning(
                            "No labels file provided and default labels do not match the number of classes."
                        )
                        boxdecoder_data["labels"] = [
                            f"Class {i}" for i in range(args.num_classes)
                        ]

                args.input_width = new_input_width
                args.input_height = new_input_height

                with open(
                    Path(plugins_dir) / "genericboxdecode" / "cfg" / "boxdecoder.json",
                    "w",
                ) as boxdecoder:
                    json.dump(obj=boxdecoder_data, fp=boxdecoder, indent=4)

            mpk_json_file_path = glob(
                f"{args.pipeline_name}/resources/*.json", recursive=True
            )[0]
            if args.no_box_decode:
                logging.info("No box decode is set, so not updating pcie_buffer_size")
                pcie_buffer_size = self.calculate_buffer_for_pcie_sink(
                    inp_depth=boxdecoder_data["input_depth"],
                    inp_width=boxdecoder_data["input_width"],
                    inp_height=boxdecoder_data["input_height"],
                )
            else:
                logging.info("Box decode is set, so updating pcie_buffer_size")
                # based on the topk from boxdecoder.json
                pcie_buffer_size = int(4 + 26*args.topk)

            args.pcie_buffer_size = pcie_buffer_size

            with open(mpk_json_file_path, 'r') as mpk_json:
                mpk_json_data = json.load(mpk_json)

            mpk_json_data["extra_automation_params"] = dict()
            mpk_json_data["extra_automation_params"]["input_width"] = new_input_width
            mpk_json_data["extra_automation_params"]["input_height"] = new_input_height
            mpk_json_data["extra_automation_params"]["model_input_width"] = model_input_width
            mpk_json_data["extra_automation_params"]["model_input_height"] = model_input_height
            mpk_json_data["extra_automation_params"][
                "detection_threshold"
            ] = args.detection_threshold
            mpk_json_data["extra_automation_params"][
                "nms_iou_threshold"
            ] = args.nms_iou_threshold
            mpk_json_data["extra_automation_params"]["num_classes"] = args.num_classes
            mpk_json_data["extra_automation_params"]["topk"] = args.topk
            mpk_json_data["extra_automation_params"]["output_buffers_size"] = int(4 + 24*args.topk)
            mpk_json_data["extra_automation_params"]["labels"] = boxdecoder_data["labels"]
            mpk_json_data["extra_automation_params"]["pcie_buffer_size"] = pcie_buffer_size
            mpk_json_data["extra_automation_params"]["model_type"] = args.model_type

            with open(mpk_json_file_path, "w") as mpk_json:
                json.dump(obj=mpk_json_data, fp=mpk_json, indent=4)

    def update_gst_commands(self, args: argparse.Namespace) -> None:
        """Update GStreamer commands in the project.

        Args:
            args (argparse.Namespace): Command line arguments.
        """
        if args.no_box_decode:
            postproc_block = f"! simaaiprocesscvu name=simaai_detesdequant_1 config=/data/simaai/applications/{args.pipeline_name}/etc/0_postproc.json "
        else:
            postproc_block = f"! simaaiboxdecode name='simaai_boxdecode' config=/data/simaai/applications/{args.pipeline_name}/etc/boxdecoder.json "

        rtsp_src = args.rtsp_src if args.rtsp_src else "<RTSP_SRC>"
        host_ip = args.host_ip if args.host_ip else "<HOST_IP>"
        host_port = args.host_port if args.host_port else "<HOST_PORT>"
        gst_cmd_rtsp = (
            f"rtspsrc location={rtsp_src} "
            "! rtph264depay wait-for-keyframe=true "
            "! h264parse "
            "! 'video/x-h264, parsed=true, stream-format=(string)byte-stream, alignment=(string)au, width=(int)[1,4096], height=(int)[1,4096]' "
            "! simaaidecoder sima-allocator-type=2 name='decoder' next-element='CVU' "
            "! tee name=source "
            "! 'video/x-raw' "
            f"! simaaiprocesscvu name=simaai_preprocess num-buffers=5 config=/data/simaai/applications/{args.pipeline_name}/etc/0_preproc.json "
            f"! simaaiprocessmla name=simaai_process_mla num-buffers=5 config=/data/simaai/applications/{args.pipeline_name}/etc/0_process_mla.json "
            f"! simaaiboxdecode name='simaai_boxdecode' config=/data/simaai/applications/{args.pipeline_name}/etc/boxdecoder.json "
            "! 'application/vnd.simaai.tensor' "
            "! overlay. source. "
            "! 'video/x-raw' "
            f"! simaai-overlay2 name=overlay render-info='input::decoder,bbox::simaai_boxdecode' labels-file='/data/simaai/applications/{args.pipeline_name}/share/overlay/labels' "
            "! simaaiencoder enc-bitrate=4000 name=encoder "
            "! h264parse "
            "! rtph264pay "
            f"! udpsink host={host_ip} port={host_port}"
            ""
        )
        gst_cmd_pcie = (
            f"simaaipciesrc buffer-size={args.input_width * args.input_height * 3} "
            f"! simaaiprocesscvu name=simaai_preprocess num-buffers=5 config=/data/simaai/applications/{args.pipeline_name}/etc/0_preproc.json "
            f"! simaaiprocessmla name=simaai_process_mla num-buffers=5 config=/data/simaai/applications/{args.pipeline_name}/etc/0_process_mla.json "
            f"{postproc_block}"
            f"!  'application/vnd.simaai.tensor' ! simaaipciesink data-buffer-size={args.pcie_buffer_size}"
        )
        logging.info(f"GStreamer commands updated succesfully for pciesink buffer size. - {args.pcie_buffer_size}")

        application_json_path = Path(args.pipeline_name) / "application.json"
        application_json_rtsp_path = Path(args.pipeline_name) / "application_rtsp.json"
        application_json_pcie_path = Path(args.pipeline_name) / "application_pcie.json"

        with open(application_json_path, "r") as file:
            application_json = json.load(file)

        logging.info("Updating application.json for multi device type structure")
        application_json = self.modify_application_json(
            application_json=application_json
        )

        # Update GStreamer commands for RTSP
        application_json["pipelines"][0]["gst"] = gst_cmd_rtsp
        with open(application_json_rtsp_path, "w+") as file:
            json.dump(application_json, file, indent=4)
            

        # Create default application JSON file
        with open(application_json_path, "w+") as file:
            if args.rtsp_src:
                # If RTSP source is provided, use the RTSP GStreamer command
                logging.info(f'Using RTSP source for GStreamer command.{application_json["pipelines"][0]["gst"]}')
                json.dump(application_json, file, indent=4)
            else:
                # If no RTSP source is provided, use the PCIe GStreamer command
                application_json_pcie = application_json.copy()
                application_json_pcie = copy.deepcopy(application_json)
                application_json_pcie["pipelines"][0]["gst"] = gst_cmd_pcie
                print("Using PCIe source for GStreamer command.")
                json.dump(application_json_pcie, file, indent=4)

        application_json_pcie = application_json.copy()
        application_json_pcie["pipelines"][0]["gst"] = gst_cmd_pcie
        with open(application_json_pcie_path, "w+") as file:
            json.dump(application_json_pcie, file, indent=4)

        logging.info("GStreamer commands updated successfully.")

    def modify_application_json(self, application_json: dict) -> dict:
        plugins_to_modify = ["detesdequant", "gen_preproc", "process_mla"]
        for plugin in application_json.get("pipelines")[0].get("plugins"):
            if plugin.get("pluginName") in plugins_to_modify:
                plugin["resources"]["configs"] = [
                    "<BOARD_TYPE>/" + plugin.get("resources").get("configs")[0]
                ]
                if "shared" in plugin.get("resources").keys():
                    plugin["resources"]["shared"] = [
                        "<BOARD_TYPE>/" + plugin.get("resources").get("shared")[0]
                    ]

        return application_json
