from pathlib import Path

import numpy as np
import typer
from PIL import Image
from streaming import MDSWriter, StreamingDataset
from torch.utils.data import DataLoader


def create_data(path_to_save: Path = Path("mds-dataset"), size: int = 100_000):
    columns = {"image": "jpeg", "class": "int"}
    compression = "zstd"

    with MDSWriter(
        out=str(path_to_save), columns=columns, compression=compression
    ) as out:
        for _ in range(size):
            sample = {
                "image": Image.fromarray(
                    np.random.randint(0, 256, (32, 32, 3), np.uint8)
                ),
                "class": np.random.randint(10),
            }
            out.write(sample)


def get_dataloader(
    remote: str = "s3://datasets/random-data", local_cache: Path = ("cache")
):
    dataset = StreamingDataset(local=str(local_cache), remote=remote, shuffle=True)
    print(dataset)
    sample = dataset[42]
    print(sample["image"], sample["class"])
    dataloader = DataLoader(dataset)
    print(f"PyTorch DataLoader = {dataloader}")


if __name__ == "__main__":
    app = typer.Typer()
    app.command()(create_data)
    app.command()(get_dataloader)
    app()
