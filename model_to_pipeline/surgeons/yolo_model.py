from pathlib import Path
import onnx
import numpy as np
import sima_utils.onnx.onnx_helpers as oh

class YOLOModel:
    def __init__(self, onnx_path):
        self.onnx_path = Path(onnx_path)
        self.model_name = self.onnx_path.stem
        self.model = oh.load_model(str(self.onnx_path))
        self.input_node_name = self._get_input_node()
        self.H, self.W, self.num_classes, self.one2one_prefix = self._infer_hw_classes()
        self.version = self._infer_model_version()
        flavor_and_splits = self._infer_model_flavor()
        if flavor_and_splits:
            self.flavor, self.splits = flavor_and_splits
        else:
            self.flavor = None
            self.splits = None
        self.has_attention = self.version in [10, 11]

    def _get_input_node(self):
        inputs = []
        initializer_names = {init.name for init in self.model.graph.initializer}
        for inp in self.model.graph.input:
            if inp.name not in initializer_names:
                inputs.append(inp.name)
        if not inputs:
            raise ValueError("No valid input node found in the model.")
        return inputs

    def _find_prefix(self, substring):
        """
        Find the prefix for nodes whose name contains a given substring.

        It scans through all nodes in the model's graph, checking if `substring`
        is in the node name. If a node's name contains at least two slashes,
        it extracts the prefix up to (and including) the second slash. 
        If none is found, it returns None.

        Args:
            model (onnx.ModelProto): Loaded ONNX model.
            substring (str): Substring to search for in node names (e.g. "m.0/Conv").

        Returns:
            str: A prefix string extracted from one of the matching node names.

        """
        prefixes = set()
        for node in self.model.graph.node:
            if substring in node.name:
                parts = node.name.split("/")
                if len(parts) >= 3:
                    # e.g. "model.121/m.0/Conv" => "model.121/m.0/Conv"
                    prefix = "/".join(parts[:3])
                    prefixes.add(prefix)
                else:
                    # If there aren't enough slashes, just use the entire name
                    prefixes.add(node.name)

        if not prefixes:
            # Gather the names of all nodes that contain the substring, for debugging
            all_matching = [node.name for node in self.model.graph.node if substring in node.name]
            # raise ValueError(
            #     f"No node found with substring '{substring}' in its name.\n"
            #     f"Matching node names found: {all_matching}"
            # )
            return None
        return prefixes.pop()
    
    def _infer_hw_classes(self):
        input_tensor = next((inp for inp in self.model.graph.input if inp.name not in {init.name for init in self.model.graph.initializer}), None)
        one2one_prefix = ""

        if input_tensor is None:
            raise ValueError("No valid input tensor found.")
        shape = [dim.dim_value for dim in input_tensor.type.tensor_type.shape.dim]
        if len(shape) != 4:
            raise ValueError("Expected 4D input (NCHW)")
        _, _, H, W = shape

        op_prefix="cv2.2/cv2.2.2/Conv"
        model_prefix = self._find_prefix(op_prefix)
        if not model_prefix:
            op_prefix="one2one_cv2.2/one2one_cv2.2.2/Conv"
            model_prefix = self._find_prefix(op_prefix)   
            one2one_prefix = "one2one_"

        if not model_prefix:
            raise ValueError("Bbox head not found in model architecture!") 

        if "cv2.2" in model_prefix:
            model_prefix = model_prefix.removesuffix(f'/{one2one_prefix}cv2.2')

        class_init = oh.find_initializer_value(self.model, f"{model_prefix[1:]}.{one2one_prefix}cv3.2.2.weight")
        return H, W, class_init.shape[0], one2one_prefix

    def _infer_model_version(self):
        if self._find_prefix("m/m.0/attn/qkv/conv/Conv"):
            return 11
        elif self._find_prefix("attn/qkv/conv/Conv"):
            return 10
        else:
            return 8

    def _infer_model_flavor(self):

        if self.version == 8:
            return None

        elif self.version == 11:

            op_prefix="attn/qkv/conv/Conv"
            model_prefix = self._find_prefix(op_prefix)
            if not model_prefix:
                return None

            model_prefix = model_prefix.removesuffix('/m')
            qkv_init = oh.find_initializer_value(self.model, f"{model_prefix[1:]}.m.0.attn.qkv.conv.weight")
            hidden_dim = qkv_init.shape[0]

            num_blocks = 2 if self._find_prefix("m.1/attn") else 1
            if hidden_dim == 256:
                return 'n', 128*np.ones(2)
            elif hidden_dim == 512:
                if num_blocks == 1:
                    return 's/m', 128*np.ones(4)
                else: 
                    return 'l', 128*np.ones(4)
            elif hidden_dim == 768:
                return 'x', 128*np.ones(6)
            else:
                raise ValueError("Cannot recognize model architecture of yolov11") 
                    
        elif self.version == 10:
            op_prefix="attn/qkv/conv/Conv"
            model_prefix = self._find_prefix(op_prefix)
            if not model_prefix:
                return None

            model_prefix = model_prefix.removesuffix('/attn')
            qkv_init = oh.find_initializer_value(self.model, f"{model_prefix[1:]}.attn.qkv.conv.weight")
            hidden_dim = qkv_init.shape[0]

            if hidden_dim == 256:
                return 'n', 128*np.ones(2)
            elif hidden_dim == 512:
                return 's/b/l', 128*np.ones(4)
            elif hidden_dim == 576:
                return 'm', 144*np.ones(4)
            elif hidden_dim == 640:
                return 'x', 128*np.ones(5)
            else:
                raise ValueError("Cannot recognize model architecture of yolov10") 
        return ValueError("Surgery not supported for this YOLO version")
        
