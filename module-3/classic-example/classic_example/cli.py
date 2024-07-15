import typer

from classic_example.data import load_sst2_data, load_sst2_data_file_input
from classic_example.train import train
from classic_example.utils import load_from_registry, upload_to_registry
from classic_example.predictor import run_inference_on_dataframe

app = typer.Typer()
app.command()(train)
app.command()(load_sst2_data)
app.command()(load_sst2_data_file_input)
app.command()(upload_to_registry)
app.command()(load_from_registry)
app.command()(run_inference_on_dataframe)


if __name__ == "__main__":
    app()
