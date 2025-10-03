#########################################################
# Copyright (C) 2024 SiMa Technologies, Inc.
#
# This material is SiMa proprietary and confidential.
#
# This material may not be copied or distributed without
# the express prior written permission of SiMa.
#
# All rights reserved.
#########################################################
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass

import numpy as np
import onnx
import onnx.version_converter
import onnxruntime as ort
from onnxsim import model_info, simplify


# Versions supported by Model SDK
_ONNX_IR_VERSION = 8
_ONNX_OPSET_VERSION = 17


@dataclass
class ModelSplit:
    """
    Configuration about how to split a model.

    :param split_name: Name of the extracted model.
    :param input_names: List of input names demarcating the beginning of the split.
    :param output_names: List of output names demarcating the end of the split.
    :param filename: File name of the extracted model to be saved.
    :param parent_split: ModelSplit for the parent if the parent is a split.
        Cannot be None when parent_path is None.
    :param parent_path: Path to the parent model to extract from.
    """
    split_name: str
    input_names: List[str]
    output_names: List[str]
    filename: Optional[Path] = None
    parent_split: Optional["ModelSplit"] = None
    parent_path: Optional[Path] = None

    def __post_init__(self):
        if self.parent_path is None:
            assert self.parent_split is not None, "parent_split cannot be None when parent_path is None"
            self.parent_path = self.parent_split.filename
        if self.filename is None:
            self.filename = Path(self.parent_path.stem + f"_{self.split_name}.onnx")

    def extract_from_parent_model(self):
        assert self.parent_path is not None
        assert self.filename is not None
        onnx.utils.extract_model(str(self.parent_path), str(self.filename), self.input_names, self.output_names)
        print(f'Extracted {self.split_name} ONNX file saved to: {str(self.filename)}')

    def run(self, input_data: List[np.ndarray]) -> List[np.ndarray]:
        assert len(input_data) == len(self.input_names)
        outputs = run_model(self.filename, self.input_names, input_data)
        assert len(outputs) == len(self.output_names)
        return outputs


#############################
# Load, save, and run models
#############################


def update_model_version(model: onnx.ModelProto) -> onnx.ModelProto:
    """
    Update an ONNX model with IR and OPSET supported by Model SDK.

    :param model: Loaded ONNX model in onnx.ModelProto representation.
    :return: Updated ONNX model in onnx.ModelProto representation.
    """
    model.ir_version = _ONNX_IR_VERSION
    return onnx.version_converter.convert_version(model, _ONNX_OPSET_VERSION)


def remove_infer_shape(model: onnx.ModelProto):
    """
    Remove shape inference results in a model.

    :param model: Loaded model in onnx.ModelProto representation.
    """
    # Remove existing shape inference results.
    for value_info in list(model.graph.value_info):
        model.graph.value_info.remove(value_info)


def load_model(model_fname: str, load_only: bool = False) -> onnx.ModelProto:
    """
    Load a model and update its version information.

    :param model_fname: File name of the model to load from disk.
    :param load_only: Boolean flag, if set to False, to simplify the model after loading.
    :return: Loaded model in onnx.ModelProto representation.
    """
    model = onnx.load(model_fname)
    model = update_model_version(model)

    if not load_only:
        model_opt, _ = simplify(model)
        model_info.print_simplifying_info(model, model_opt)
        model = model_opt
    return model


def save_model(model: onnx.ModelProto, model_fname: str, save_only: bool = False):
    """
    Save a model to disk.

    :param model: Model to be saved.
    :param model_fname: File name to be used to save the model.
    :param save_only: Boolean flag, if set to False, to simplify and re-generate shape inference
        result before saving to disk.
    """
    model = update_model_version(model)
    if not save_only:
        # Simplify model.
        model_opt, check = simplify(model)
        assert check, "Simplified ONNX model can not be validated"
        model_info.print_simplifying_info(model, model_opt)
        model_opt = onnx.shape_inference.infer_shapes(model_opt)
        onnx.checker.check_model(model_opt, full_check=True)
        model = model_opt
    onnx.save(model, model_fname)
    print(f'ONNX file saved to {model_fname}')


def run_model(model_name: str, input_names: List[str], input_data: List[np.ndarray]) -> List[np.ndarray]:
    """
    Run inference on a model.

    :param model_name: File name of the model saved on disk.
    :param input_names: List of input names expected by the model.
    :param input_data: List of input data for the model to run inference on.
    :return: List of outputs from inference result.
    """
    assert len(input_names) == len(input_data)
    sess = ort.InferenceSession(str(model_name))
    outputs = sess.run([], {name: np_data for name, np_data in zip(input_names, input_data)})
    return outputs


def verify_models_equal(model_0: str, model_1: str, input_names: List[str], input_data: List[np.ndarray]):
    """
    Run inference on two models and expect identical match numerically.

    :param model_0: File name of the first model.
    :param model_1: File name of the second model.
    :param input_names: List of input names expected by both models.
    :param input_data: List of input data to run inference on.
    """
    outputs_0 = run_model(model_0, input_names, input_data)
    outputs_1 = run_model(model_1, input_names, input_data)
    assert np.array_equal(outputs_0, outputs_1)
    print(f"Verification OK - {model_0} and {model_1} produce same outputs")


def verify_models_close(model_0: str, model_1: str, input_names: List[str], input_data: List[np.ndarray],
                        atol: float = 1e-06):
    """
    Run inference on two models and expect close match numerically.

    :param model_0: File name of the first model.
    :param model_1: File name of the second model.
    :param input_names: List of input names expected by both models.
    :param input_data: List of input data to run inference on.
    :param atol: Float value as absolute tolerance in comparison.
    """
    outputs_0 = run_model(model_0, input_names, input_data)
    outputs_1 = run_model(model_1, input_names, input_data)
    assert np.allclose(outputs_0, outputs_1, atol=atol)
    print(f"Verification OK - {model_0} and {model_1} are close within {atol}")


#############################
# Split and merge models
#############################


def extract_model(model: onnx.ModelProto, input_names: List[str], output_names: List[str]) -> onnx.ModelProto:
    """
    Extract from a model.

    :param model: The model to extract from.
    :param input_names: List of input names demarcating the beginning of the split.
    :param output_names: List of output names demarcating the end of the split.
    :return: Extracted model in onnx.ModelProto representation.
    """
    return onnx.utils.Extractor(model).extract_model(input_names, output_names)


def split_model(model_splits: List[ModelSplit]):
    """
    Split a model.

    :param model_splits: List of ModelSplit to perform model extraction.
    """
    for ms in model_splits:
        ms.extract_from_parent_model()


def verify_split_models(whole_model: str, split_models: List[ModelSplit],
                        input_names: List[str], input_data: List[np.ndarray]):
    """
    Verify split models by comparing final inference outputs of the whole model and the cascaded splits.
        The order of the splits must be in the same execution order of the original whole model.

    :param whole_model: File name of the original whole model.
    :param split_models: List of ModelSplit extracted from the whole model.
    :param input_names: List of input names expected by the whole model.
    :param input_data: List of input data to run inference on.
    """
    ref_outputs = run_model(whole_model, input_names, input_data)

    inputs = input_data
    for ms in split_models:
        outputs = ms.run(inputs)
        inputs = outputs

    assert np.array_equal(ref_outputs, outputs)
    print(f"Verification OK - {len(split_models)} split models are equivalent to {whole_model}")
    print([ms.split_name for ms in split_models])


def merge_model(model_0: Union[str, Path, onnx.ModelProto], model_1: Union[str, Path, onnx.ModelProto],
                io_map: List[Tuple[str, str]]) -> onnx.ModelProto:
    """
    Merge two models.

    :param model_0: First model provided by file name or loaded as onnx.ModelProto.
    :param model_1: Second model provided by file name or loaded as onnx.ModelProto.
    :param io_map: List of tuples mapping output of the first model to input of the second model.
    :return: Merged model in onnx.ModelProto representation.
    """
    # Load the models if needed
    if isinstance(model_0, (str, Path)):
        model_0 = onnx.load(model_0)

    if isinstance(model_1, (str, Path)):
        model_1 = onnx.load(model_1)

    # Merge the models
    num_outputs_added = 0
    for model_1_input in model_1.graph.input:
        if model_1_input in model_0.graph.input:
            model_0.graph.output.append(
                onnx.helper.make_tensor_value_info(
                    model_1_input.name,
                    onnx.TensorProto.FLOAT,
                    get_io_shape(model_1_input)
                )
            )
            num_outputs_added += 1
            io_map.append((model_1_input.name, model_1_input.name))

    model_out = onnx.compose.merge_models(model_0, model_1, io_map)

    for _ in range(num_outputs_added):
        model_0.graph.output.pop()

    return model_out


def merge_split_model_with_shared_constant(base_model: Union[onnx.ModelProto, None], model_split: str,
                                           constant_prefix: str = None) -> onnx.ModelProto:
    """
    Merge two models which may have shared constants.
        When a model is split into multiple parts, some constants are duplicated in multiple parts.
        Later, when they are merged (possibly after modifications), there are conflicts because of
        the same constant names. A solution is to add prefix to constant names in the second model.

    :param model: Loaded model in onnx.ModelProto representation.
    :param model_split: File name of the second model to merge.
    :param constant_prefix: Prefix name to prepend to all constants in the second model.
    :return: Merged model in onnx.ModelProto representation.
    """
    # Load the second model
    split_model = onnx.load(model_split)
    if base_model is None:
        print(f"---------Merge: = {model_split}")
        return split_model

    # Rename overlapping constant names in the second model
    if constant_prefix:
        split_model = uniquify_initializer_name(constant_prefix, split_model)

    # Get O/I map as list of pairs of string, [(out0, in0), (out1, in1), …]
    # representing outputs of the first graph and inputs of the second to be connected
    base_outputs = [node.name for node in base_model.graph.output]
    new_inputs = [node.name for node in split_model.graph.input]
    new_initializers = [ini.name for ini in split_model.graph.initializer]
    input_to = list(set(new_inputs) - set(new_initializers))
    assert len(base_outputs) == len(input_to), \
        f"Cannot merge - input to {model_split} not match output of the base model"
    oi_map = list(zip(base_outputs, input_to))

    # Remove existing shape inference results from the model to be added
    for value_info in list(split_model.graph.value_info):
        split_model.graph.value_info.remove(value_info)

    new_model = onnx.compose.merge_models(base_model, split_model, oi_map)
    print(f"---------Merge OK: {model_split}")
    return new_model


#############################
# Operator nodes (generic)
#############################


def find_node(model: onnx.ModelProto, node_name: str) -> onnx.NodeProto:
    """
    Find an operator node by name.

    :param model: Loaded model in onnx.ModelProto representation.
    :param node_name: Name of a node.
    :return: Operator node in onnx.NodeProto representation.
    """
    for node in model.graph.node:
        if node.name == node_name:
            return node
    raise RuntimeError(f"Node {node_name} not found")


def find_node_output(model: onnx.ModelProto, output_name: str) -> onnx.NodeProto:
    """
    Find an operator node whose output is the specified output_name.

    :param model: Loaded model in onnx.ModelProto representation.
    :param output_name: Name of the output.
    :return: Operator node in onnx.NodeProto representation.
    """
    for node in list(model.graph.node):
        if node.output[0] == output_name:
            return node
    raise RuntimeError(f"Node with output as {output_name} not found")


def find_node_input(model: onnx.ModelProto, input_name: str) -> Tuple[onnx.NodeProto, int]:
    """
    Find an operator node whose input is the specified input_name.

    :param model: Loaded model in onnx.ModelProto representation.
    :param input_name: Name of the input.
    :return: Tuple of Operator node and input index.
    """
    for node in list(model.graph.node):
        for i, input in enumerate(node.input):
            if input == input_name:
                return node, i
    raise RuntimeError(f"Node with input as {input_name} not found")


def make_node(**kwargs: Any) -> onnx.NodeProto:
    """
    Make an operator node following ONNX specification.

    :param **kwargs: Operator dependent properties.
    :return: Newly constructed node in onnx.NodeProto representation.
    """
    return onnx.helper.make_node(**kwargs)


def remove_node(model: onnx.ModelProto, node: Union[str, onnx.NodeProto], remove_only: bool = False):
    """
    Remove a node in a model.

    :param model: Loaded model in onnx.ModelProto representation.
    :param node: Node to be removed, provided as node name or onnx.NodeProto representation.
    :param remove_only: Boolean flag, if set to False, to reconnect surrounding nodes.
    """
    if isinstance(node, str):
        node = find_node(model, node)

    if not remove_only:
        # Check if the removed node is a last node
        assert len(node.output) == 1
        is_last_node = any(node.output[0] == x.name for x in list(model.graph.output))

        # Find true input to the node (not an initializer)
        true_input_to_removed_node = [name for name in node.input if not is_initializer(model, name)]
        assert len(true_input_to_removed_node) == 1

        # If the removed node is a last node, reuse the output name for the connecting node.
        # Otherwise, update the input name of the following node.
        if is_last_node:
            # Update output name of the connecting node
            connecting_node = find_node_output(model, true_input_to_removed_node[0])
            connecting_node.output[0] = node.output[0]            
        else:
            # Update input name of the following node
            following_node, input_idx = find_node_input(model, node.output[0])
            following_node.input[input_idx] = true_input_to_removed_node[0]

    model.graph.node.remove(node)


def insert_node(model: onnx.ModelProto, node: onnx.NodeProto, new_node: onnx.NodeProto, 
                insert_before: bool = False, insert_only: bool = False):
    """
    Insert a node in a model.

    :param model: Loaded model in onnx.ModelProto representation.
    :param node: Reference node where to insert a new node.
    :param new_node: New node to be inserted.
    :param insert_before: Boolean flag, True to insert before or False to insert after the reference node.
    :param insert_only: Boolean flag, if False, to connect the new node to the reference node.
    """
    if isinstance(node, str):
        node = find_node(model, node)
    for i, x in enumerate(list(model.graph.node)):
        if x.name != node.name:
            continue

        if insert_before:
            if not insert_only:
                new_node.input[0] = x.input[0]
                x.input[0] = new_node.output[0]
            model.graph.node.insert(i, new_node)
        else:
            if not insert_only:
                new_node.output[0] = x.output[0]
                x.output[0] = new_node.input[0]
            model.graph.node.insert(i + 1, new_node)
        return


def replace_node(model: onnx.ModelProto, node: onnx.NodeProto, new_nodes: List[onnx.NodeProto]):
    """
    Replace a node in a model.

    :param model: Loaded model in onnx.ModelProto representation.
    :param node: Reference node to replace.
    :param new_nodes: List of nodes to replace with.
    """
    for i, x in enumerate(list(model.graph.node)):
        if x.name != node.name:
            continue
        model.graph.node.remove(node)

        for j, new_node in enumerate(new_nodes):
            model.graph.node.insert(i + j, new_node)
        return


def set_attribute_to_node(model: onnx.ModelProto, node_name: str, attr_name: str, val: Any):
    """
    Set attribute of a node in a model.

    :param model: Loaded model in onnx.ModelProto representation.
    :param node_name: Name of the node.
    :param attr_name: Name of the attribute.
    :param val: Value of the attribute.
    """
    for idx, node in enumerate(model.graph.node):
        if node.name == node_name:
            attr = onnx.helper.make_attribute(attr_name, val)
            node.attribute[0].CopyFrom(attr)


def remove_nodes_by_name_list(model: onnx.ModelProto, name_list: List[str]):
    """
    Remove multiple nodes by a list of names.

    :param model: Loaded model in onnx.ModelProto representation.
    :param name_list: List of names of the nodes to be removed.
    """
    for idx, node in enumerate(model.graph.node):
        if node.name in name_list:
            model.graph.node.remove(node)


#############################
# Operator nodes (specific)
#############################


def insert_transpose_pair(model: onnx.ModelProto, node_name: str,
                          perm_before: Tuple[int], perm_after: Tuple[int]) -> List[str]:
    """
    Insert a pair of transpose operators before and after a node.

    :param model: Loaded model in onnx.ModelProto representation.
    :param node_name: Name of the reference node.
    :param perm_before: Permutation before the node.
    :param perm_after: Permutation after the node.
    :return: List of names of the newly inserted transpose nodes.
    """
    new_nodes = []
    for node_idx, node in enumerate(model.graph.node):
        if node.name == node_name:
            new_node_name = node_name+"/Transpose_0"
            tr_before = onnx.helper.make_node(
                op_type="Transpose",
                name=new_node_name, 
                inputs=[node.input[0]],
                outputs=[node_name+"/Transpose_0_output"],
                perm=perm_before
            )
            new_nodes.append(new_node_name)
            node.input[0] = tr_before.output[0]
            new_node_name = node_name+"/Transpose_1"
            tr_after = onnx.helper.make_node(
                op_type="Transpose",
                name=new_node_name, 
                inputs=node.output,
                outputs=[node_name+"/Transpose_1_output"],
                perm=perm_after
            )
            new_nodes.append(new_node_name)
            model.graph.node.insert(node_idx, tr_before)
            model.graph.node.insert(node_idx+2, tr_after)
            break
    return new_nodes


def rewrite_matmul_as_conv(model: onnx.ModelProto, name_list: List[str]) -> List[str]:
    """
    Rewrite a Matmul operator as a Conv operator.

    :param model: Loaded model in onnx.ModelProto representation.
    :param name_list: List of Matmul node names to be rewritten.
    :return: List of names of the new convolution nodes.
    """
    new_nodes = []
    for idx, node in enumerate(model.graph.node):
        if node.name in name_list:
            assert node.op_type == "MatMul"
            model.graph.node.remove(node)
            new_node_name = node.name + "_Conv"
            model.graph.node.insert(
                idx,
                onnx.helper.make_node(
                    op_type="Conv",
                    name=new_node_name, 
                    inputs=[node.input[0], node.input[1]],
                    outputs=node.output,
                    kernel_shape=(1, 1),
                    group=1
                )
            )
            new_nodes.append(new_node_name)
            # Weights: Transpose and Reshape matrix
            for initializer in list(model.graph.initializer):
                if initializer.name == node.input[1]:
                    data = onnx.numpy_helper.to_array(initializer)
                    data = data.transpose([1, 0])
                    data = np.reshape(data, newshape=(*data.shape, 1, 1))
                    initializer.CopyFrom(onnx.numpy_helper.from_array(data, initializer.name))

    return new_nodes


def rewrite_matmul_as_einsum(model: onnx.ModelProto, eqn_list: Dict[str, str]) -> List[str]:
    """
    Rewrite a Matmul operator as an Einsum operator.

    :param model: Loaded model in onnx.ModelProto representation.
    :param eqn_list: Dictionary for rewrite with key being node name and value being equation string for Einsum.
    :return: List of names of the new Einsum nodes.
    """
    new_nodes = []
    for idx, node in enumerate(model.graph.node):
        if node.name in eqn_list.keys():
            assert node.op_type == "MatMul"
            model.graph.node.remove(node)
            new_node_name = node.name + "_Einsum"
            model.graph.node.insert(
                idx,
                onnx.helper.make_node(
                    op_type="Einsum",
                    name=new_node_name,
                    inputs=node.input,
                    outputs=node.output,
                    equation=eqn_list[node.name]
                )
            )
            new_nodes.append(new_node_name)

    return new_nodes


def insert_slices_after_node(model: onnx.ModelProto, node_name: str, *,
                             axis: int, nslices: int, slice_size: int) -> List[str]:
    """
    Insert Slice operators after a node. All Slice operators output the same length on the slicing axis.

    :param model: Loaded model in onnx.ModelProto representation.
    :param node_name: Name of the node after which to insert slices.
    :param axis: Axis for the Slice operators.
    :param nslices: Number of Slice operators.
    :param slice_size: Length for the Slice operators on the slicing axis.
    :return: List of names of the new Slice nodes.
    """
    # Slice axis is a shared constant
    slice_axes = node_name + ":slice_axes"
    model.graph.initializer.append(
        onnx.helper.make_tensor(slice_axes, onnx.TensorProto.INT64, [1], [axis])
    )
    inserted_nodes = []
    for idx, node in enumerate(model.graph.node):
        if node.name == node_name:
            for n in range(nslices):
                slice_starts = node_name + f":slice_{n}_starts"
                slice_ends = node_name + f":slice_{n}_ends"
                model.graph.initializer.append(
                    onnx.helper.make_tensor(slice_starts, onnx.TensorProto.INT64, [1], [n*slice_size])
                )
                model.graph.initializer.append(
                    onnx.helper.make_tensor(slice_ends, onnx.TensorProto.INT64, [1], [(n+1)*slice_size])
                )
                new_node_name = node.name + f"/Slice_{n}"
                slice_node = onnx.helper.make_node(
                    op_type="Slice", 
                    name=new_node_name,
                    inputs=[node.output[0], slice_starts, slice_ends, slice_axes],
                    outputs=[node.name + f"/Slice_{n}_output"]
                )
                model.graph.node.insert(idx+1+n, slice_node)
                inserted_nodes.append(new_node_name)

    return inserted_nodes


def insert_concat(model: onnx.ModelProto, input_nodes: List[str], at_node: str, *, axis: int) ->  str:
    """
    Insert Concat operator after a group of nodes.

    :param model: Loaded model in onnx.ModelProto representation.
    :param input_nodes: List of names representing the group of nodes which are inputs to Concat.
    :param at_node: Name of node where insertion happens.
    :param axis: Axis on which to concatenate.
    :return: Name of newly inserted Concat node.
    """
    # Collect inputs
    split_outputs = []
    for idx, node in enumerate(model.graph.node):
        if node.name in input_nodes:
            split_outputs.append(node.output[0])
    # Insert concat
    for idx, node in enumerate(model.graph.node):
        if node.name == at_node:
            new_node_name = node.name + "_Concat"
            model.graph.node.insert(
                idx,
                onnx.helper.make_node(
                    op_type="Concat",
                    name=new_node_name, 
                    inputs=split_outputs,
                    outputs=[node.name + "_Concat_output"],
                    axis=axis
                )
            )
            break
    return new_node_name


def insert_slices_concat_between_nodes(model: onnx.ModelProto, after_node: str, before_node: str,
                                       *,
                                       slice_axis: int, nslices: int, slice_size: int,
                                       concat_axis: int) -> str:
    """
    Insert Slices + Concat between two nodes.

    :param model: Loaded model in onnx.ModelProto representation.
    :param after_node: Name of the node after which a group of Slice operators are inserted.
    :param before_node: Name of the node before which a Concat operator is inserted.
    :param slice_axis: Axis for the Slice operators.
    :param nslices: Number of Slice operators.
    :param slice_size: Length for the Slice operators on the slicing axis.
    :param concat_axis: Axis on which to concatenate.
    :return: Name of newly inserted Concat node.
    """
    # First, insert slices
    slice_nodes = insert_slices_after_node(model, after_node, axis=slice_axis, nslices=nslices, slice_size=slice_size)
    # Then, insert a concat
    new_node = insert_concat(model, slice_nodes, before_node, axis=concat_axis)
    return new_node


def rewrite_gemm_as_conv(model: onnx.ModelProto, after_node: str, at_node: str, *, 
                         w_r: int, w_s: int, w_c: int, w_k: int) -> str:
    """
    Rewrite a Gemm operator as a Conv operator between two nodes.

    :param model: Loaded model in onnx.ModelProto representation.
    :param after_node: Name of the node after which the Conv operator is inserted.
    :param at_node: Name of the node before which the Conv operator is inserted.
    :param w_r, w_s, w_c, w_k: Shape of the convolution weight as RSCK
    :return: Name of newly inserted Conv node.
    """
    for idx, node in enumerate(model.graph.node):
        if node.name == at_node:
            assert node.op_type == "Gemm"
            model.graph.node.remove(node)
            new_node_name = node.name + "_Conv"
            model.graph.node.insert(
                idx,
                onnx.helper.make_node(
                    op_type="Conv",
                    name=new_node_name,
                    inputs=[find_node(model, after_node).output[0], node.input[1], node.input[2]],
                    outputs=node.output,
                    kernel_shape=(w_r, w_s),
                )
            )
            # Weight
            def weight_convert_func(data):
                return data.reshape(w_k, w_c, w_r, w_s)
            
            convert_initializer(model, node.input[1], "custom", convert_func=weight_convert_func)

    return new_node_name


def connect_nodes(model: onnx.ModelProto, node_pair: List[str], out_idx: int, in_idx: int):
    """
    Connect two nodes by feeding the output of one node to the input of next node.

    :param model: Loaded model in onnx.ModelProto representation.
    :param node_pair: List of two nodes in execution order.
    :param out_idx: The index of output of the first node.
    :param in_idx: The index of input of the second node.
    """
    node_1 = find_node(model, node_pair[0])
    node_2 = find_node(model, node_pair[1])
    node_2.input[in_idx] = node_1.output[out_idx]


#############################
# Inputs and outputs
#############################


def remove_output(model: onnx.ModelProto, keep_list: List[str] = None):
    """
    Run outputs of a model.

    :param model: Loaded model in onnx.ModelProto representation.
    :param keep_list: List of output names not to be removed.
    """
    for o in list(model.graph.output):
        if keep_list and o.name in keep_list:
            continue
        model.graph.output.remove(o)


def remove_outputs_by_name_list(model: onnx.ModelProto, name_list: List[str]):
    """
    Remove multiple outputs by a list of names.

    :param model: Loaded model in onnx.ModelProto representation.
    :param name_list: List of names of the outputs to be removed.
    """
    for o in list(model.graph.output):
        if o.name in name_list:
            model.graph.output.remove(o)


def remove_inputs_by_name_list(model: onnx.ModelProto, name_list: List[str]):
    """
    Remove multiple inputs by a list of names.

    :param model: Loaded model in onnx.ModelProto representation.
    :param name_list: List of names of the inputs to be removed.
    """
    for i in list(model.graph.input):
        if i.name in name_list:
            model.graph.input.remove(i)


def add_io(model: onnx.ModelProto, io_name: str, io_shape: Tuple[int], io_dir: str):
    """
    Add input or output to a model.

    :param model: Loaded model in onnx.ModelProto representation.
    :param io_name: Name to be added for an input/output.
    :param io_shape: Shape of the added input/output.
    :param io_dir: Designation of input or output.
    """
    tensor_value_info = onnx.helper.make_tensor_value_info(
        io_name,
        onnx.TensorProto.FLOAT,
        io_shape
    )
    if io_dir == "in":
        model.graph.input.append(tensor_value_info)
    else:
        assert io_dir == "out"
        model.graph.output.append(tensor_value_info)


def add_input(model: onnx.ModelProto, input_name: str, input_shape: Tuple[int]):
    """
    Add input to a model.

    :param model: Loaded model in onnx.ModelProto representation.
    :param input_name: Name of the input to be added.
    :param input_shape: Shape of the input to be added.
    """
    add_io(model, input_name, input_shape, "in")


def add_output(model: onnx.ModelProto, output_name, output_shape):
    """
    Add output to a model.

    :param model: Loaded model in onnx.ModelProto representation.
    :param output_name: Name of the output to be added.
    :param output_shape: Shape of the output to be added.
    """
    add_io(model, output_name, output_shape, "out")


def get_io_shape(node: onnx.NodeProto) -> Tuple[int]:
    """
    Get the shape of a tensor.

    :param node: Node representing the tensor.
    :return: Shape of the tensor.
    """
    shape = [d.dim_value for d in node.type.tensor_type.shape.dim]
    return tuple(shape)


def update_io_shape(model: onnx.ModelProto, io_name: str, new_shape: Tuple[int]):
    """
    Update I/O shape in a model.

    :param model: Loaded model in onnx.ModelProto representation.
    :param io_name: Name of the I/O
    :param new_shape: Shape of the I/O
    """
    def _update_shape(node):
        node.type.CopyFrom(
            onnx.helper.make_tensor_type_proto(
                onnx.TensorProto.FLOAT,
                new_shape
            )
        )

    for node in model.graph.input:
        if node.name == io_name:
            _update_shape(node)
            return

    for node in model.graph.output:
        if node.name == io_name:
            _update_shape(node)
            return

    raise RuntimeError(f"IO name not found in model: {io_name}")


def change_node_output(model: onnx.ModelProto, node_name: str, new_output_name: str, old_output_name: str = None):
    """
    Change output of a node in a model.

    :param model: Loaded model in onnx.ModelProto representation.
    :param node_name: Name of a node.
    :param new_output_name: Name of the new output for the node.
    :param old_output_name: Name of previous output to replace.
    """
    node = find_node(model, node_name)

    for idx, o in enumerate(node.output):
        if old_output_name is None and idx == 0:
            node.output[0] = new_output_name
            return
        if o == old_output_name:
            node.output[idx] = new_output_name
            return


#############################
# Initializers
#############################


def is_initializer(model: onnx.ModelProto, name: str) -> bool:
    """
    Check if a name is an initializer.

    :param model: Loaded model in onnx.ModelProto representation.
    :param name: Name to check on.
    :return: True if provided name is an initializer.
    """
    for initializer in model.graph.initializer:
        if initializer.name == name:
            return True
    return False


def find_initializer_value(model: onnx.ModelProto, initializer_name: str) -> np.ndarray:
    """
    Find value of an initializer in a model.

    :param model: Loaded model in onnx.ModelProto representation.
    :param initializer_name: Name of the initializer.
    :return: Value of the initializer.
    """
    found = False
    for initializer in model.graph.initializer:
        if initializer.name == initializer_name:
            found = True
            break
    if found:
        return onnx.numpy_helper.to_array(initializer)
    else:
        raise RuntimeError(f"Initializer {initializer_name} not found")


def remove_initializers(model: onnx.ModelProto, name_list: List[str]):
    """
    Remove initializers from a list.

    :param model: Loaded model in onnx.ModelProto representation.
    :param name_list: List of initializer names.
    """
    for initializer in list(model.graph.initializer):
        if initializer.name in name_list:
            model.graph.initializer.remove(initializer)


def add_initializer(model: onnx.ModelProto, initializer_name: str, initializer_value: np.ndarray):
    """
    Add an initializer to a model.

    :param model: Loaded model in onnx.ModelProto representation.
    :param initializer_name: Name of the initializer.
    :param initializer_value: Value of the initializer.
    """
    model.graph.initializer.append(
        onnx.helper.make_tensor(
            name=initializer_name,
            data_type=onnx.TensorProto.FLOAT,
            dims=initializer_value.shape,
            vals=initializer_value.flatten().tolist()
        )
    )


def remove_duplicated_initializer(model: onnx.ModelProto):
    """
    Remove duplicated initializers in a model.

    :param model: Loaded model in onnx.ModelProto representation.
    """
    unique_inits = dict()
    for init in list(model.graph.initializer):
        data = find_initializer_value(model, init)
        if init.name in unique_inits:
            assert np.array_equal(unique_inits[init.name], data)
            model.graph.initializer.remove(init)
        else:
            unique_inits[init.name] = data


def transpose_reshape_constant(model: onnx.ModelProto, param_name: str, perm: Tuple[int], new_shape: Tuple[int]):
    """
    Perform transpose and/or reshape on an existing constant initializer.

    :param model: Loaded model in onnx.ModelProto representation.
    :param param_name: Name of the constant as an initializer.
    :param perm: Permutation to be performed.
    :param new_shape: New shape to be reshaped.
    """
    for initializer in list(model.graph.initializer):
        if initializer.name == param_name:
            data = onnx.numpy_helper.to_array(initializer)
            if perm:
                data = data.transpose(perm)
            data = np.reshape(data, newshape=new_shape)
            initializer.CopyFrom(onnx.numpy_helper.from_array(data, initializer.name))


def convert_initializer(model: onnx.ModelProto, init_name: str, convert_type: str,
                        convert_func: Optional[Callable[[np.ndarray], np.ndarray]] = None):
    """
    Convert an initializer in a model.

    :param model: Loaded model in onnx.ModelProto representation.
    :param init_name: Name of the initializer.
    :param convert_type: String describing format conversion.
    :param convert_func: Custom conversion function for "custom" convert_type.
    """
    for initializer in model.graph.initializer:
        if initializer.name == init_name:
            data = onnx.numpy_helper.to_array(initializer)
            if convert_type == "wbc_to_chw":
                data = data.transpose(1, 2, 0)[:, :, np.newaxis, :][0]
            elif convert_type == "ck_to_kchw":
                data = data.transpose(1, 0)[:, :, np.newaxis, np.newaxis]
            elif convert_type == "xc_to_xchw":
                data = data[..., np.newaxis, np.newaxis]
            elif convert_type == "hwc_to_nchw":
                data = data.transpose(2, 0, 1)[np.newaxis, ...]
            elif convert_type == "custom":
                assert convert_func is not None
                data = convert_func(data)
            initializer.CopyFrom(onnx.numpy_helper.from_array(data, initializer.name))
            return


def uniquify_initializer_name(prefix_name: str, model: onnx.ModelProto) -> onnx.ModelProto:
    """
    Uniquify initializer names in a model by a prefix name.

    :param prefix_name: Name of the prefix to prepend to all initializer names.
    :param model: Loaded model in onnx.ModelProto representation.
    :return: Updated model in onnx.ModelProto representation.
    """
    return onnx.compose.add_prefix(
                model,
                prefix_name + "/",
                rename_nodes=False,
                rename_edges=False,
                rename_inputs=False,
                rename_outputs=False,
                rename_initializers=True,
                rename_value_infos=False,
                rename_functions=False,
                inplace=True)
