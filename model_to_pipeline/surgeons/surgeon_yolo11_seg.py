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
import os
import onnx
from onnxsim import simplify

import model_to_pipeline.utils.onnx_helpers as oh

import numpy as np
from model_to_pipeline.surgeons.surgeon_base import SurgeonBase
from model_to_pipeline.surgeons.yolo_model import YOLOModel
import onnxruntime as ort


    
def get_node_names(model):
    inputs = []
    initializer_names = {init.name for init in model.graph.initializer}
    for inp in model.graph.input:
        if inp.name not in initializer_names:
            inputs.append(inp.name)
    if not inputs:
        raise ValueError("No valid input node found in the model.")
    return inputs

def remove_unused_initializers(model):

    used_inputs = set()
    for node in model.graph.node:
        for name in node.input:
            used_inputs.add(name)

    new_initializers = []
    removed_initializers = []
    for init in model.graph.initializer:
        if init.name in used_inputs:
            new_initializers.append(init)
        else:
            removed_initializers.append(init.name)

    print("Removing unused initializers:")
    for name in removed_initializers:
        print("  ", name)

    del model.graph.initializer[:]
    model.graph.initializer.extend(new_initializers)

    new_inputs = []
    for inp in model.graph.input:
        if inp.name in used_inputs or any(inp.name == init.name for init in model.graph.initializer):
            new_inputs.append(inp)
    del model.graph.input[:]
    model.graph.input.extend(new_inputs)

    return model

def run_inference(model_path, input_data, output_nodes=None):
    session = ort.InferenceSession(model_path)
    input_name = session.get_inputs()[0].name
    return session.run(output_nodes, {input_name: input_data}) if output_nodes else session.run(None, {input_name: input_data})

def compare_tensors(tensor1, tensor2, atol=0.00001):
    if tensor1.shape != tensor2.shape:
        print(f"different shapes: {tensor1.shape} vs {tensor2.shape}")
        return
    difference_mask = ~np.isclose(tensor1, tensor2, atol=atol)
    if np.any(difference_mask):
        print("difference found at indexes: ")
        for index in np.argwhere(difference_mask):
            idx = tuple(index)
            print(f"ind {idx}: tensor1={tensor1[idx]}, tensor2={tensor2[idx]}")
    else:
        print(f"tensors are close with atol {atol}")

def reshape_sima_output(output, num_classes):
    for k in range(6):
        output[k] = np.transpose(output[k], (0,2,3,1))
    pred_bbox = np.concatenate([output[k].reshape(1, -1, 4) for k in range(3)], axis=1)
    pred_prob = np.concatenate([output[k].reshape(1, -1, num_classes) for k in range(3, 6)], axis=1)
    return np.concatenate([pred_bbox, pred_prob], axis=2).transpose(0, 2, 1)


class SurgeonYolo11Seg(SurgeonBase):
    name = "yolo11-seg"

    def do_surgery(self, args: argparse.Namespace) -> str:
        """
        Perform the surgery on the YOLOv8 model.
        This method should be overridden by subclasses.
        """
        assert os.path.exists(args.model_path), "Model path does not exist."
        model_name = os.path.splitext(os.path.basename(args.model_path))[0]

        onnx_model = args.model_path
        args.post_surgery_model_path = args.post_surgery_model_path if args.post_surgery_model_path else args.pipeline_name + "_mod.onnx"
        # Extract the model parameters
        # Load the ONNX model
        model = oh.load_model(onnx_model)

        # Extract input and output shapes
        input_shapes = {
            inp.name: tuple(dim.dim_value for dim in inp.type.tensor_type.shape.dim)
            for inp in model.graph.input
        }
        output_shapes = {
            out.name: tuple(dim.dim_value for dim in out.type.tensor_type.shape.dim)
            for out in model.graph.output
        }

        print("Input Shapes:", input_shapes)
        print("Output Shapes:", output_shapes)

        # Extract width and height from the input shape
        input_shape = list(input_shapes.values())[
            0
        ]  # Assuming the first input shape is the one we need
        height, width = input_shape[2], input_shape[3]
        print(f"Input Height: {height}, Input Width: {width}")

        # Extract number of classes from the output shape
        output_shape = list(output_shapes.values())[
            0
        ]  # Assuming the first output shape is the one we need
        num_classes = output_shape[2] - 4 - 32
        print(f"Number of Classes: {num_classes}")

        imgz_size = (height, width)

        mod_model_name = f"{model_name}_opt"

        # automate the number of classes and input size
        H, W = imgz_size

        model = oh.load_model(f"{model_name}.onnx", load_only=True)

        print("inside surgeon_yolo11.py")

        model_path = f"{model_name}.onnx"
        yolo = YOLOModel(model_path)
        model = yolo.model
        model_name = yolo.model_name
        mod_model_name = f"{model_name}_mod"
        input_node_name = yolo.input_node_name
        model_prefix_no_block = yolo._find_prefix("attn/qkv/conv/Conv") 
        H, W = yolo.H, yolo.W
        num_classes = yolo.num_classes
        one2one_prefix = yolo.one2one_prefix
        model_flavor = yolo.flavor
        splits = yolo.splits
        has_attn = yolo.has_attention

        print(f"Model: {model_name}, Version: yolov{yolo.version}, Flavor/size: {model_flavor}, H: {H}, W: {W}, Classes: {num_classes}")

        if has_attn:
            for block in range(2 if (yolo.version==11 and model_flavor in ['l', 'x']) else 1): # 'yolov11x' has 2 attn blocks

                if yolo.version != 10:
                    model_prefix = f"{model_prefix_no_block}/m.{block}/attn"
                else:
                    model_prefix = model_prefix_no_block
                #Matmul replacement

                matmul1 =f"{model_prefix}/MatMul"
                matmul2 = f"{model_prefix}/MatMul_1" #einsum shape inference does not work, so the output may be unknown or wrong on the graph

                dict1 = {
                matmul1: "nchw,nchq->nqhw",
                }
                dict2 = {
                    matmul2: "nchw,nqhc->nqhw"
                }
                oh.rewrite_matmul_as_einsum(model = model, eqn_list= dict1)
                oh.rewrite_matmul_as_einsum(model = model, eqn_list= dict2)

                # replace first reshape
                initializer_0 = onnx.helper.make_tensor(
                        name=f'block_{block}',
                        data_type=onnx.TensorProto.INT64,
                        dims=[H//32],
                        vals=np.ones(H//32).astype(np.int64)
                    )
                model.graph.initializer.extend([initializer_0])

                conv = oh.find_node(model,f'{model_prefix}/qkv/conv/Conv')
                oh.remove_node(model, f"{model_prefix}/Reshape",True)
                oh.insert_node(
                    model,
                    conv,
                    oh.make_node(
                        name=f"/block_{block}_split_resh",
                        op_type="Split",
                        inputs=[conv.output[0],f"block_{block}"],
                        outputs=[f"/block_{block}_split_output{i+1}" for i in range(H//32)],
                        axis=2,
                    ),
                    insert_only=True,
                )
                split2=oh.find_node(model,f'{model_prefix}/Split')
                oh.insert_node(
                    model,
                    split2,
                    oh.make_node(
                        name=f"/block_{block}_concat1_resh",
                        op_type="Concat",
                        inputs=[f"/block_{block}_split_output{i+1}" for i in range(H//32)],
                        outputs=[f"/block_{block}_concat1_output"],
                        axis=3,
                    ),
                    insert_only=True,
                    insert_before=True
                )

                num_128s = len(splits)
                vals = splits

                conc=oh.find_node(model,f'/block_{block}_concat1_resh')
                # replace first reshape
                initializer_0 = onnx.helper.make_tensor(
                        name=f'block_{block}2',
                        data_type=onnx.TensorProto.INT64,
                        dims=[num_128s],
                        vals=np.array(vals, dtype=np.int64)
                    )
                model.graph.initializer.extend([initializer_0])

                oh.insert_node(
                    model,
                    conc,
                    oh.make_node(
                        name=f"/block_{block}_split2_resh",
                        op_type="Split",
                        inputs=[conc.output[0],f"block_{block}2"],
                        outputs=[f"/block_{block}_split2_output{i+1}" for i in range(num_128s)],
                        axis=1,
                    ),
                    insert_only=True,
                )
                split2=oh.find_node(model,f'{model_prefix}/Split')
                oh.insert_node(
                    model,
                    split2,
                    oh.make_node(
                        name=f"/block_{block}_concat2_resh1",
                        op_type="Concat",
                        inputs=[f"/block_{block}_split2_output{i+1}" for i in range(num_128s)],
                        outputs=[f"/block_{block}_concat2_output"],
                        axis=2,
                    ),
                    insert_only=True,
                    insert_before=True
                )

                vals = [32,32,64] if not (yolo.version == 10 and yolo.flavor =='m') else [36,36,72]

                conc=oh.find_node(model,f'/block_{block}_concat2_resh1')
                split2.input[0]=conc.output[0]
                oh.remove_node(model,f"{model_prefix}/Split",True)

                initializer_0 = onnx.helper.make_tensor(
                        name=f'block_{block}3',
                        data_type=onnx.TensorProto.INT64,
                        dims=[3],
                        vals=np.array(vals, dtype=np.int64)
                    )
                model.graph.initializer.extend([initializer_0])

                oh.insert_node(
                    model,
                    conc,
                    oh.make_node(
                        name=f"/block_{block}_Split_to_3_branches",
                        op_type="Split",
                        inputs=[conc.output[0],f"block_{block}3"],
                        outputs=[f"/block_{block}_Split_to_3_branches_output1",
                    f"/block_{block}_Split_to_3_branches_output2", f"/block_{block}_Split_to_3_branches_output3"],
                        axis=1,
                    ),
                    insert_only=True,
                )
                Split_to_3_branches=oh.find_node(model,f"/block_{block}_Split_to_3_branches")
                out1=oh.find_node(model,f"{model_prefix}/Transpose")
                out2=oh.find_node(model,f"{model_prefix}/MatMul_Einsum")
                out3=oh.find_node(model,f"{model_prefix}/MatMul_1_Einsum")
                out4=oh.find_node(model,f"{model_prefix}/Reshape_2")
                out1.input[0],out2.input[1],out3.input[0],out4.input[0]=Split_to_3_branches.output[0],Split_to_3_branches.output[1],Split_to_3_branches.output[2],Split_to_3_branches.output[2]
                oh.remove_node(model,f"{model_prefix}/Transpose")
                eins=oh.find_node(model,f"{model_prefix}/MatMul_Einsum")
                eins.input[0]=Split_to_3_branches.output[0]
                oh.remove_node(model,f"{model_prefix}/Softmax",True)
                mul=oh.find_node(model,f"{model_prefix}/Mul")
                mul.input[0]=eins.output[0]
                oh.insert_node(
                    model,
                    mul,
                    oh.make_node(
                        name=f"/block_{block}_softmax_resh",
                        op_type="Softmax",
                        inputs=[mul.output[0]],
                        outputs=[f"/block_{block}_softmax_output"],
                        axis=1,
                    ),
                    insert_only=True,
                )
                soft=oh.find_node(model,f"/block_{block}_softmax_resh")
                transp=oh.find_node(model,f"{model_prefix}/Transpose_1")
                transp.input[0]=soft.output[0]
                oh.remove_node(model,f"{model_prefix}/Transpose_1")
                eins=oh.find_node(model,f"{model_prefix}/MatMul_1_Einsum")
                eins.input[0],eins.input[1]=soft.output[0],Split_to_3_branches.output[2]

                # replace second reshape
                oh.remove_node(model,f"{model_prefix}/Reshape_1",True)

                initializer_0 = onnx.helper.make_tensor(
                        name=f'block_{block}_splittwo_tens',
                        data_type=onnx.TensorProto.INT64,
                        dims=[num_128s],
                        vals=np.array(np.ones(num_128s), dtype=np.int64)
                    )
                model.graph.initializer.extend([initializer_0])

                oh.insert_node(
                    model,
                    eins,
                    oh.make_node(
                        name=f"/block_{block}_split_reshtwo",
                        op_type="Split",
                        inputs=[eins.output[0],f"block_{block}_splittwo_tens"],
                        outputs=[f"/block_{block}_splittwo_output{i+1}" for i in range(num_128s)],
                        axis=2,
                    ),
                    insert_only=True,
                )
                split2=oh.find_node(model,f'/block_{block}_split_reshtwo')
                oh.insert_node(
                    model,
                    split2,
                    oh.make_node(
                        name=f"/block_{block}_concat_reshtwo",
                        op_type="Concat",
                        inputs=[f"/block_{block}_splittwo_output{i+1}" for i in range(num_128s)],
                        outputs=[f"/block_{block}_concattwo_output"],
                        axis=1,
                    ),
                    insert_only=True,
                )
                add=oh.find_node(model,f"{model_prefix}/Add")
                conc=oh.find_node(model,f"/block_{block}_concat_reshtwo")

                initializer_0 = onnx.helper.make_tensor(
                        name=f'block_{block}_splittwo_tens2',
                        data_type=onnx.TensorProto.INT64,
                        dims=[H//32],
                        vals=W//32 * np.ones(H//32).astype(np.int64)
                    )
                model.graph.initializer.extend([initializer_0])

                oh.insert_node(
                    model,
                    conc,
                    oh.make_node(
                        name=f"/block_{block}_splittwo_resh2",
                        op_type="Split",
                        inputs=[conc.output[0],f"block_{block}_splittwo_tens2"],
                        outputs=[f"/block_{block}_splittwo1_output{i+1}" for i in range(H//32)],
                        axis=3,
                    ),
                    insert_only=True,
                )
                split2=oh.find_node(model,f'/block_{block}_splittwo_resh2')
                oh.insert_node(
                    model,
                    split2,
                    oh.make_node(
                        name=f"/block_{block}_concattwo_resh2",
                        op_type="Concat",
                        inputs=[f"/block_{block}_splittwo1_output{i+1}" for i in range(H//32)],
                        outputs=[f"/block_{block}_concattwo_output1"],
                        axis=2,
                    ),
                    insert_only=True,
                )
                conc=oh.find_node(model,f'/block_{block}_concattwo_resh2')
                add.input[0]=conc.output[0]

                # replace third reshape
                conv=oh.find_node(model,f"{model_prefix}/pe/conv/Conv")
                oh.remove_node(model,f"{model_prefix}/Reshape_2",True)

                initializer_0 = onnx.helper.make_tensor(
                        name=f'block_{block}_splittwo2_tens',
                        data_type=onnx.TensorProto.INT64,
                        dims=[num_128s],
                        vals=np.array(np.ones(num_128s), dtype=np.int64)
                    )
                model.graph.initializer.extend([initializer_0])

                oh.insert_node(
                    model,
                    conv,
                    oh.make_node(
                        name=f"/block_{block}_split2_reshtwo",
                        op_type="Split",
                        inputs=[Split_to_3_branches.output[2],f"block_{block}_splittwo2_tens"],
                        outputs=[f"/block_{block}_splittwo2_output{i+1}" for i in range(num_128s)],
                        axis=2,
                    ),
                    insert_only=True,
                    insert_before=True
                )
                split2=oh.find_node(model,f'/block_{block}_split2_reshtwo')
                oh.insert_node(
                    model,
                    split2,
                    oh.make_node(
                        name=f"/block_{block}_concat_reshtwo2",
                        op_type="Concat",
                        inputs=[f"/block_{block}_splittwo2_output{i+1}" for i in range(num_128s)],
                        outputs=[f"/block_{block}_concattwo2_output"],
                        axis=1,
                    ),
                    insert_only=True,
                )
                add=oh.find_node(model,f"{model_prefix}/Add")
                conc=oh.find_node(model,f"/block_{block}_concat_reshtwo2")

                initializer_0 = onnx.helper.make_tensor(
                        name=f'block_{block}_splittwo22_tens2',
                        data_type=onnx.TensorProto.INT64,
                        dims=[H//32],
                        vals=W//32*np.ones(H//32).astype(np.int64)
                    )
                model.graph.initializer.extend([initializer_0])

                oh.insert_node(
                    model,
                    conc,
                    oh.make_node(
                        name=f"/block_{block}_splittwo2_resh2",
                        op_type="Split",
                        inputs=[conc.output[0],f"block_{block}_splittwo22_tens2"],
                        outputs=[f"/block_{block}_splittwo22_output{i+1}" for i in range(H//32)],
                        axis=3,
                    ),
                    insert_only=True,
                )
                split2=oh.find_node(model,f'/block_{block}_splittwo2_resh2')
                oh.insert_node(
                    model,
                    split2,
                    oh.make_node(
                        name=f"/block_{block}_concattwo2_resh2",
                        op_type="Concat",
                        inputs=[f"/block_{block}_splittwo22_output{i+1}" for i in range(H//32)],
                        outputs=[f"/block_{block}_concattwo2_output1"],
                        axis=2,
                    ),
                    insert_only=True,
                )
                conc=oh.find_node(model,f'/block_{block}_concattwo2_resh2')
                conv.input[0]=conc.output[0]

        onnx.save(model, f"{mod_model_name}.onnx")

        model_opt, check = simplify(model)
        assert check, "Simplified ONNX model can not be validated"
        model = model_opt

        oh.remove_infer_shape(model)
        model = onnx.shape_inference.infer_shapes(model)
        onnx.checker.check_model(model, full_check=True)
        onnx.save(model, f"{mod_model_name}.onnx")

        input_onnx = f"{mod_model_name}.onnx"
        output_onnx = f"{model_name}_opt.onnx"

        model = onnx.load(input_onnx)

        # Remove all outputs and reconstruct outputs.
        oh.remove_output(model)
        oh.add_output(model, "bbox_0", (1, 4, H//8, W//8))
        oh.add_output(model, "bbox_1", (1, 4, H//16, W//16))
        oh.add_output(model, "bbox_2", (1, 4, H//32, W//32))
        oh.add_output(model, "class_prob_0", (1, num_classes, H//8, W//8))
        oh.add_output(model, "class_prob_1", (1, num_classes, H//16, W//16))
        oh.add_output(model, "class_prob_2", (1, num_classes, H//32, W//32))


        op_prefix=f"{one2one_prefix}cv2.0/{one2one_prefix}cv2.0.2/Conv"
        model_prefix = yolo._find_prefix(op_prefix)
        if model_prefix:
            if f"{one2one_prefix}cv2.0" in model_prefix:
                model_prefix = model_prefix.removesuffix(f'/{one2one_prefix}cv2.0')

        is_seg = False

        op_prefix=f"{one2one_prefix}cv4.0/{one2one_prefix}cv4.0.2/Conv"
        seg_prefix = yolo._find_prefix(op_prefix)
        if seg_prefix:
            if f"{one2one_prefix}cv4.0" in seg_prefix:
                seg_prefix = seg_prefix.removesuffix(f'/{one2one_prefix}cv4.0')
        if seg_prefix:
            is_seg = True
            oh.add_output(model, "mask_coeff_0", (1, 32, H//8, W//8))
            oh.add_output(model, "mask_coeff_1", (1, 32, H//16, W//16))
            oh.add_output(model, "mask_coeff_2", (1, 32, H//32, W//32))
            oh.add_output(model, "mask", (1, 32, H//4, W//4))
            print("This is a segmentation model")
            # Modify mask path.
            oh.change_node_output(model, f"{seg_prefix}/proto/cv3/act/Mul", "mask")

            # Modify mask coeff path.
            oh.change_node_output(model, f"{seg_prefix}/cv4.0/cv4.0.2/Conv", "mask_coeff_0")
            oh.change_node_output(model, f"{seg_prefix}/cv4.1/cv4.1.2/Conv", "mask_coeff_1")
            oh.change_node_output(model, f"{seg_prefix}/cv4.2/cv4.2.2/Conv", "mask_coeff_2")
            oh.remove_node(model, f"{seg_prefix}/Reshape", True)
            oh.remove_node(model, f"{seg_prefix}/Reshape_1", True)
            oh.remove_node(model, f"{seg_prefix}/Reshape_2", True)
            oh.remove_node(model, f"{seg_prefix}/Concat", True)
        else:
            print("This is a detection only model")

        # Modify bbox path.
        bbox_version = 2
        addsub_const = oh.find_initializer_value(model, f"{model_prefix}/Constant_{12 if is_seg else 9}_output_0")
        mul_const = oh.find_initializer_value(model, f"{model_prefix}/Constant_{14 if is_seg else 12}_output_0")
        cur_off = 0
        for conv_idx in range(3):
            base_name = f"{model_prefix}/{one2one_prefix}cv2.{conv_idx}/{one2one_prefix}cv2.{conv_idx}.2"
            old_conv_name = f"{base_name}/Conv"
            old_conv_node = oh.find_node(model, old_conv_name)

            old_conv_weight = oh.find_initializer_value(model, old_conv_node.input[1])
            old_conv_bias = oh.find_initializer_value(model, old_conv_node.input[2])

            mul_name = f"{model_prefix}/{one2one_prefix}cv2.{conv_idx}/{one2one_prefix}cv2.{conv_idx}.1/act/Mul"
            mul_node = oh.find_node(model, mul_name)

            dfl_conv_nodes = [None]*4
            for split_idx in range(3, -1, -1):
                new_conv_name = f"{base_name}/{split_idx}/Conv"
                new_conv_weight_name = f"{new_conv_name}.weight"
                new_conv_bias_name = f"{new_conv_name}.bias"

                oh.add_initializer(model, new_conv_weight_name, old_conv_weight[16*split_idx:16*(split_idx+1), ...])
                oh.add_initializer(model, new_conv_bias_name, old_conv_bias[16*split_idx:16*(split_idx+1)])

                oh.insert_node(
                    model,
                    mul_node,
                    new_conv_node := oh.make_node(
                        name=new_conv_name,
                        op_type="Conv",
                        inputs=[mul_node.output[0], new_conv_weight_name, new_conv_bias_name],
                        outputs=[f"{new_conv_name}_output"]
                    ),
                    insert_only=True
                )

                new_base_name = f"{model_prefix}/dfl/{conv_idx}/{split_idx}"
                new_softmax_name = f"{new_base_name}/Softmax"
                oh.insert_node(
                    model,
                    new_conv_node,
                    new_softmax_node := oh.make_node(
                        name=new_softmax_name,
                        op_type="Softmax",
                        inputs=new_conv_node.output,
                        outputs=[f"{new_softmax_name}_output"],
                        axis=1
                    ),
                    insert_only=True
                )

                new_conv_name = f"{new_base_name}/Conv"
                oh.insert_node(
                    model,
                    new_softmax_node,
                    new_conv_node := oh.make_node(
                        name=new_conv_name,
                        op_type="Conv",
                        inputs=[new_softmax_node.output[0], "model.23.dfl.conv.weight"],
                        outputs=[f"{new_conv_name}_output"]
                    ),
                    insert_only=True
                )
                dfl_conv_nodes[split_idx] = new_conv_node

            cur_h = H//(2**(conv_idx+3))
            cur_w = W//(2**(conv_idx+3))
            if bbox_version == 1:
                new_base_name = f"{model_prefix}/dfl/{conv_idx}"
                new_concat_name = f"{new_base_name}/Concat_0"
                oh.insert_node(
                    model,
                    dfl_conv_nodes[3],
                    concat_0_node := oh.make_node(
                        name=new_concat_name,
                        op_type="Concat",
                        inputs=[dfl_conv_nodes[0].output[0], dfl_conv_nodes[1].output[0]],
                        outputs=[f"{new_concat_name}_output"],
                        axis=1
                    ),
                    insert_only=True
                )
                new_concat_name = f"{new_base_name}/Concat_1"
                oh.insert_node(
                    model,
                    concat_0_node,
                    concat_1_node := oh.make_node(
                        name=new_concat_name,
                        op_type="Concat",
                        inputs=[dfl_conv_nodes[2].output[0], dfl_conv_nodes[3].output[0]],
                        outputs=[f"{new_concat_name}_output"],
                        axis=1
                    ),
                    insert_only=True
                )

                cur_addsub_const = addsub_const[..., cur_off:cur_off+cur_h*cur_w].reshape(1, 2, cur_h, cur_w)
                cur_mul_const = mul_const[..., cur_off:cur_off+cur_h*cur_w].reshape(1, cur_h, cur_w)
                cur_off += cur_h*cur_w

                new_sub_name = f"{new_base_name}/Sub_0"
                oh.add_initializer(model, f"{new_sub_name}/Const", cur_addsub_const)

                oh.insert_node(
                    model,
                    concat_1_node,
                    sub_0_node := oh.make_node(
                        name=new_sub_name,
                        op_type="Sub",
                        inputs=[f"{new_sub_name}/Const", concat_0_node.output[0]],
                        outputs=[f"{new_sub_name}_output"]
                    ),
                    insert_only=True
                )

                new_add_name = f"{new_base_name}/Add_0"
                oh.add_initializer(model, f"{new_add_name}/Const", cur_addsub_const)

                oh.insert_node(
                    model,
                    sub_0_node,
                    add_0_node := oh.make_node(
                        name=new_add_name,
                        op_type="Add",
                        inputs=[f"{new_add_name}/Const", concat_1_node.output[0]],
                        outputs=[f"{new_add_name}_output"]
                    ),
                    insert_only=True
                )

                new_add_name = f"{new_base_name}/Add_1"
                oh.insert_node(
                    model,
                    add_0_node,
                    add_1_node := oh.make_node(
                        name=new_add_name,
                        op_type="Add",
                        inputs=[sub_0_node.output[0], add_0_node.output[0]],
                        outputs=[f"{new_add_name}_output"]
                    ),
                    insert_only=True
                )

                new_div_name = f"{new_base_name}/Div"
                oh.insert_node(
                    model,
                    add_1_node,
                    div_node := oh.make_node(
                        name=new_div_name,
                        op_type="Div",
                        inputs=[add_1_node.output[0], f"{model_prefix}/Constant_24_output_0"],
                        outputs=[f"{new_div_name}_output"]
                    ),
                    insert_only=True
                )

                new_sub_name = f"{new_base_name}/Sub_1"
                oh.insert_node(
                    model,
                    div_node,
                    sub_1_node := oh.make_node(
                        name=new_sub_name,
                        op_type="Sub",
                        inputs=[add_0_node.output[0], sub_0_node.output[0]],
                        outputs=[f"{new_sub_name}_output"]
                    ),
                    insert_only=True
                )

                new_concat_name = f"{new_base_name}/Concat_2"
                oh.insert_node(
                    model,
                    sub_1_node,
                    concat_node := oh.make_node(
                        name=new_concat_name,
                        op_type="Concat",
                        inputs=[div_node.output[0], sub_1_node.output[0]],
                        outputs=[f"{new_concat_name}_output"],
                        axis=1
                    ),
                    insert_only=True
                )

                new_mul_name = f"{new_base_name}/Mul"
                oh.add_initializer(model, f"{new_mul_name}/Const", cur_mul_const)
                oh.insert_node(
                    model,
                    concat_node,
                    oh.make_node(
                        name=new_mul_name,
                        op_type="Mul",
                        inputs=[concat_node.output[0], f"{new_mul_name}/Const"],
                        outputs=[f"bbox_{conv_idx}"]
                    ),
                    insert_only=True
                )
            else:
                new_base_name = f"{model_prefix}/dfl/{conv_idx}"
                new_concat_name = f"{new_base_name}/Concat"
                oh.insert_node(
                    model,
                    dfl_conv_nodes[3],
                    concat_node := oh.make_node(
                        name=new_concat_name,
                        op_type="Concat",
                        inputs=[x.output[0] for x in dfl_conv_nodes],
                        outputs=[f"{new_concat_name}_output"],
                        axis=1
                    ),
                    insert_only=True
                )

                cur_mul_const = 2**(conv_idx+3)
                new_conv_name = f"{new_base_name}/Conv"
                conv_weight = np.array(
                    [
                        [-0.5, 0, 0.5, 0],
                        [0, -0.5, 0, 0.5],
                        [1, 0, 1, 0],
                        [0, 1, 0, 1]
                    ],
                ).reshape(4, 4, 1, 1) * cur_mul_const
                oh.add_initializer(model, f"{new_conv_name}.weight", conv_weight)

                oh.insert_node(
                    model,
                    concat_node,
                    conv_node := oh.make_node(
                        name=new_conv_name, 
                        op_type="Conv",
                        inputs=[concat_node.output[0], f"{new_conv_name}.weight"],
                        outputs=[f"{new_conv_name}_output"]
                    ),
                    insert_only=True
                )

                new_add_name = f"{new_base_name}/Add"
                add_const = list()
                for i in range(4):
                    add_const.append(list())
                    for j in range(cur_h):
                        add_const[i].append(list())
                        for k in range(cur_w):
                            if i == 0:
                                add_const[i][j].append(0.5 + k)
                            elif i == 1:
                                add_const[i][j].append(0.5 + j)
                            else:
                                add_const[i][j].append(0)
                add_const = np.array(add_const).reshape(1, 4, cur_h, cur_w) * cur_mul_const
                oh.add_initializer(model, f"{new_add_name}/Const", add_const)
                oh.insert_node(
                    model,
                    conv_node,
                    oh.make_node(
                        name=new_add_name,
                        op_type="Add",
                        inputs=[conv_node.output[0], f"{new_add_name}/Const"],
                        outputs=[f"bbox_{conv_idx}"]
                    ),
                    insert_only=True
                )

            oh.remove_node(model, old_conv_name, True)


        # Modify class probability path.
        for conv_idx in range(3):
            base_name = f"{model_prefix}/{one2one_prefix}cv3.{conv_idx}/{one2one_prefix}cv3.{conv_idx}.2"
            conv_name = f"{base_name}/Conv"
            sigmoid_name = f"{base_name}/Sigmoid"
            conv_node = oh.find_node(model, conv_name)
            oh.insert_node(
                model,
                conv_node,
                oh.make_node(
                    name=sigmoid_name,
                    op_type="Sigmoid",
                    inputs=conv_node.output,
                    outputs=[f"class_prob_{conv_idx}"]
                ),
                insert_only=True
            )

        model = remove_unused_initializers(model)

        # Remove all unneeded nodes.
        oh.remove_node(model, f"{model_prefix}/Slice_1", True)
        oh.remove_node(model, f"{model_prefix}/Sigmoid", True)
        oh.remove_node(model, f"{model_prefix}/Concat_1", True)
        oh.remove_node(model, f"{model_prefix}/Concat_2", True)
        oh.remove_node(model, f"{model_prefix}/Concat_3", True)
        oh.remove_node(model, f"{model_prefix}/Concat_4", True)
        oh.remove_node(model, f"{model_prefix}/Slice", True)
        oh.remove_node(model, f"{model_prefix}/dfl/Reshape", True)
        oh.remove_node(model, f"{model_prefix}/dfl/Transpose", True)
        oh.remove_node(model, f"{model_prefix}/dfl/Softmax", True)
        oh.remove_node(model, f"{model_prefix}/dfl/conv/Conv", True)
        oh.remove_node(model, f"{model_prefix}/dfl/Reshape_1", True)
        oh.remove_node(model, f"{model_prefix}/Split", True)

        if yolo.version == 11:
            oh.remove_node(model, f"{model_prefix}/Add_2", True)
            oh.remove_node(model, f"{model_prefix}/Add_1", True)
            oh.remove_node(model, f"{model_prefix}/Sub", True)
            oh.remove_node(model, f"{model_prefix}/Sub_1", True)
            oh.remove_node(model, f"{model_prefix}/Div_1", True)
            oh.remove_node(model, f"{model_prefix}/Concat_5", True)
            oh.remove_node(model, f"{model_prefix}/Mul_2", True)

        else:
            oh.remove_node(model, f"{model_prefix}/Add_1", True)
            oh.remove_node(model, f"{model_prefix}/Sub", True)
            oh.remove_node(model, f"{model_prefix}/Mul_2", True)
            oh.remove_node(model, f"{model_prefix}/Concat_5", True)
            oh.remove_node(model, f"{model_prefix}/Transpose", True)
            oh.remove_node(model, f"{model_prefix}/Split_1", True)
            oh.remove_node(model, f"{model_prefix}/ReduceMax", True)
            oh.remove_node(model, f"{model_prefix}/TopK", True)
            oh.remove_node(model, f"{model_prefix}/Unsqueeze", True)
            oh.remove_node(model, f"{model_prefix}/Tile", True)
            oh.remove_node(model, f"{model_prefix}/Tile_1", True)
            oh.remove_node(model, f"{model_prefix}/GatherElements", True)
            oh.remove_node(model, f"{model_prefix}/GatherElements_1", True)
            oh.remove_node(model, f"{model_prefix}/Flatten", True)
            oh.remove_node(model, f"{model_prefix}/Flatten_1", True)
            oh.remove_node(model, f"{model_prefix}/TopK_1", True)
            oh.remove_node(model, f"{model_prefix}/Mod", True)
            oh.remove_node(model, f"{model_prefix}/Div_1", True)
            oh.remove_node(model, f"{model_prefix}/Gather_3", True)
            oh.remove_node(model, f"{model_prefix}/Unsqueeze_1", True)
            oh.remove_node(model, f"{model_prefix}/Unsqueeze_2", True)
            oh.remove_node(model, f"{model_prefix}/Cast_2", True)
            oh.remove_node(model, f"{model_prefix}/Concat_8", True)

        if is_seg:  
            oh.remove_node(model, f"{model_prefix}/Reshape_3", True)
            oh.remove_node(model, f"{model_prefix}/Reshape_4", True)
            oh.remove_node(model, f"{model_prefix}/Reshape_5", True)
            oh.remove_node(model, f"{model_prefix}/Concat_6", True)      
        else:
            oh.remove_node(model, f"{model_prefix}/Reshape", True)
            oh.remove_node(model, f"{model_prefix}/Reshape_1", True)
            oh.remove_node(model, f"{model_prefix}/Reshape_2", True)
            oh.remove_node(model, f"{model_prefix}/Concat", True)

        onnx.save(model, args.post_surgery_model_path)
        oh.save_model(model, args.post_surgery_model_path)


        input_data = np.random.uniform(low=-1.0, high=1.0, size=(1,3,H,W)).astype(np.float32)

        orig_cut = [f"{model_prefix}/cv2.2/cv2.2.1/act/Mul_output_0"]

        for i, cut_node in enumerate(orig_cut):

            model1 = oh.load_model(model_path, load_only=True)
            value_info_protos = [vi for vi in onnx.shape_inference.infer_shapes(model1).graph.value_info if vi.name == cut_node]
            model1.graph.output.extend(value_info_protos)
            onnx.checker.check_model(model1)
            inter_path1 = f'{model_name}_inter.onnx'
            onnx.save(model1, inter_path1)

            model2 = oh.load_model(model_path, load_only=True)
            value_info_protos = [vi for vi in onnx.shape_inference.infer_shapes(model2).graph.value_info if vi.name == cut_node]
            model2.graph.output.extend(value_info_protos)
            onnx.checker.check_model(model2)
            inter_path2 = f'{model_name}_opt_inter.onnx'
            onnx.save(model2, inter_path2)

            print(f"model 1: {inter_path1}")
            print(f"model 2: {inter_path2}")

            output_node_1 = run_inference(inter_path1, input_data, output_nodes=[cut_node])[0]
            output_node_2 = run_inference(inter_path2, input_data, output_nodes=[cut_node])[0]
            # output_node_2 = run_inference(f"{model_name}_opt.onnx", input_data)[i]

            print(f"Comparing output at node {i}: {cut_node}:")
            compare_tensors(output_node_1, output_node_2)

        os.remove(inter_path1)
        os.remove(inter_path2)
        os.remove(f"{mod_model_name}.onnx")

        print(f"model {model_name}_opt.onnx done\n\n")

        return
