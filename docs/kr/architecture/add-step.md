## Adding New Step
Follow this if you wish to add new step to workflow of the tool.

* Create a python script under `steps` directory

* Create a new step class and extend it with step's base class named `StepBase`. Add two class variables named `name` and `sequence` under it.
  * `name` is used to identify the step
  * `sequence` is used to sequence the steps for excution

* Override the `run` method in your custom step class

```python
class StepMyStep(StepBase):
    """Step for performing MyStep on the model.
    """

    name = "mystep"
    sequence = 4

    def run(self, args: argparse.Namespace) -> bool:
        """Run the mystep step.

        Args:
            args: Arguments passed to the step.

        Returns:
            bool: True if the mystep was successful, False otherwise.
        """
        # Step logic for your step
```
