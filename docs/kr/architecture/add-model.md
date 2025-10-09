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