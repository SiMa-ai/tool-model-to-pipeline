## Adding a new tool under `sima-model-to-pipeline`

To add a new tool altogether: <br>
1. Create a directory at root level of the project. For example, let's call it `my_new_tool`.<br>
2. Add a `__init__.py` under it so that that directory is considered as a module <br>
3. Create another python script under it which contains the main logic of your code. Make sure that you use `typer` module's cocept to make your sub-tool a part of main tool and accessible via it. You can use below template for doing so.

```python
import typer

my_cli_tool = typer.Typer()


@my_cli_tool.command()
def run(
    command_arg: str = typer.Option(
        ...,
        help="Compulsory arg",
        metavar="choice1/choice2",
        show_choices=True,
        prompt=False,
        case_sensitive=False,
    )
) -> None:
    """This my_cli_tool does some work"""
    # Your logic goes here
    typer.echo(f"Running my_cli_tool with arg: {command_arg}")
```

4. Now under `sima_tool/main.py`, import `my_cli_tool` as

```python
from my_new_tool.main import my_cli_tool
```

5. Add this imported tool to parent `typer` app in `sima_tool/main.py` as

```python
app.add_typer(fps_getter_app)
app.add_typer(model_to_pipeline_app)
app.add_typer(my_cli_tool) # ---> This is new entry
```

6. After this, your tool's entry will be shown under the main `sima-model-to-pipeline tool`.