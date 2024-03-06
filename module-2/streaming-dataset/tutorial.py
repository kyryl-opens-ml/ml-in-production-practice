import numpy as np
from PIL import Image
from streaming import MDSWriter
import typer
from torch.utils.data import DataLoader
from streaming import StreamingDataset

from pathlib import Path

def create_data(path_to_save: Path = Path('mds-dataset'), size: int = 10000):
    columns = {
        'image': 'jpeg',
        'class': 'int'
    }
    compression = 'zstd'
    
    with MDSWriter(out=str(path_to_save), columns=columns, compression=compression) as out:
        for i in range(size):
            sample = {
                'image': Image.fromarray(np.random.randint(0, 256, (32, 32, 3), np.uint8)),
                'class': np.random.randint(10),
            }
            out.write(sample)



def get_dataloader(remote: str = 's3://datasets/random-data', local_cache: Path = ('cache')):
    dataset = StreamingDataset(local=str(local_cache), remote=remote, shuffle=True)
    print(dataset)
    sample = dataset[42]
    print(sample['image'].shape, sample['class'])
    # Create PyTorch DataLoader
    dataloader = DataLoader(dataset)    

if __name__ == '__main__':
    app = typer.Typer()
    app.command()(create_data)
    app.command()(get_dataloader)
    app()