import typer

from generative_example.data import load_sql_data, load_sql_data_file_input
from generative_example.train import train
from generative_example.predictor import run_inference_on_json, run_evaluate_on_json
from generative_example.utils import load_from_registry, upload_to_registry

app = typer.Typer()
app.command()(load_sql_data)
app.command()(load_sql_data_file_input)
app.command()(train)
app.command()(upload_to_registry)
app.command()(load_from_registry)
app.command()(run_inference_on_json)
app.command()(run_evaluate_on_json)


if __name__ == "__main__":
    app()
