import numpy as np
from sklearn.dummy import DummyClassifier
import concurrent.futures
from typing import Tuple
import time
from tqdm import tqdm
from concurrent.futures import wait
import time
import typer
import ray
from dask.distributed import Client



def train_model(x_train: np.ndarray, y_train: np.ndarray) -> DummyClassifier:
    dummy_clf = DummyClassifier(strategy="most_frequent")
    dummy_clf.fit(x_train, y_train)
    return dummy_clf


def get_data(inference_size: int = 100_000_000) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    x_train = np.random.rand(100)
    y_train = np.random.rand(100)

    x_test = np.random.rand(inference_size)
    return x_train, y_train, x_test


def predict(model: DummyClassifier, x: np.ndarray) -> np.ndarray:
    # replace with real model

    # dim = 150
    # np.linalg.inv(np.random.rand(dim * dim).reshape((dim, dim)))
    
    time.sleep(0.002)
    return model.predict(x)


def run_inference(model: DummyClassifier, x_test: np.ndarray, batch_size: int = 2048) -> np.ndarray:
    y_pred = []
    y_batch = predict(model, x_test)

    for i in tqdm(range(0, x_test.shape[0], batch_size)):
        x_batch = x_test[i : i + batch_size]
        y_batch = predict(model, x_batch)
        y_pred.append(y_batch)
    return np.concatenate(y_pred)


    w: 16
    r: 1GB
    batch_size: 4000 ~ 500MB


    g: 20GB
    w: 4
    batch_size: 40000 ~ 500MB










def run_inference_process_pool(model: DummyClassifier, x_test: np.ndarray, max_workers: int = 16) -> np.ndarray:

    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        chunk_size = len(x_test) // max_workers

        chunks = []
        # split in to chunks
        for i in range(0, len(x_test), chunk_size):
            chunks.append(x_test[i : i + chunk_size])

        futures = []
        # submit chunks for inference
        for chunk in chunks:
            future = executor.submit(run_inference, model=model, x_test=chunk)
            futures.append(future)

        # wait for all futures
        wait(futures)

        y_pred = []
        for future in futures:
            y_batch = future.result()
            y_pred.append(y_batch)
    return np.concatenate(y_pred)


@ray.remote
def run_inference_ray(model: DummyClassifier, x_test: np.ndarray, batch_size: int = 2048) -> np.ndarray:
    y_pred = []
    for i in range(0, x_test.shape[0], batch_size):
        x_batch = x_test[i : i + batch_size]
        y_batch = predict(model, x_batch)
        y_pred.append(y_batch)
    return np.concatenate(y_pred)


def run_inference_ray_main(model: DummyClassifier, x_test: np.ndarray, max_workers: int = 16) -> np.ndarray:
    chunk_size = len(x_test) // max_workers

    chunks = []
    # Split into chunks
    for i in range(0, len(x_test), chunk_size):
        chunks.append(x_test[i : i + chunk_size])

    # Run inference on chunks
    futures = [run_inference_ray.remote(model, chunk) for chunk in chunks]

    # Collect results
    y_pred = ray.get(futures)
    return np.concatenate(y_pred)


def run_inference_dask(model: DummyClassifier, x_test: np.ndarray, batch_size: int = 2048) -> np.ndarray:
    y_pred = []
    for i in range(0, x_test.shape[0], batch_size):
        x_batch = x_test[i : i + batch_size]
        y_batch = predict(model, x_batch)
        y_pred.append(y_batch)
    return np.concatenate(y_pred)


def run_inference_dask_main(client, model: DummyClassifier, x_test: np.ndarray, max_workers: int = 16) -> np.ndarray:
    chunk_size = len(x_test) // max_workers

    chunks = []
    # Split into chunks
    for i in range(0, len(x_test), chunk_size):
        chunks.append(x_test[i : i + chunk_size])

    # Run inference on chunks
    futures = [client.submit(run_inference_dask, model, chunk) for chunk in chunks]

    # Collect results
    y_pred = client.gather(futures)
    return np.concatenate(y_pred)





def run_single_worker(inference_size: int = 100_000_000):

    x_train, y_train, x_test = get_data(inference_size=inference_size)

    model = train_model(x_train, y_train)

    s = time.monotonic()

    y_test_predicted = run_inference(model=model, x_test=x_test)

    print(f"Inference one worker {time.monotonic() - s} restulst: {y_test_predicted.shape}")













def run_pool(inference_size: int = 100_000_000, max_workers: int = 16):
    x_train, y_train, x_test = get_data(inference_size=inference_size)
    model = train_model(x_train, y_train)

    s = time.monotonic()
    res = run_inference_process_pool(model=model, x_test=x_test)
    print(f"Inference {max_workers} workers {time.monotonic() - s} restulst: {res.shape}")


def run_ray(inference_size: int = 100_000_000, max_workers: int = 16):
    ray.init()

    x_train, y_train, x_test = get_data(inference_size=inference_size)
    model = train_model(x_train, y_train)

    s = time.monotonic()
    res = run_inference_ray_main(model=model, x_test=x_test, max_workers=max_workers)
    print(f"Inference with Ray {time.monotonic() - s} restulst: {res.shape}")


def run_dask(inference_size: int = 100_000_000, max_workers: int = 16):
    client = Client()

    x_train, y_train, x_test = get_data(inference_size=inference_size)
    model = train_model(x_train, y_train)

    s = time.monotonic()
    res = run_inference_dask_main(client=client, model=model, x_test=x_test, max_workers=max_workers)
    print(f"Inference with Dask {time.monotonic() - s} restulst: {res.shape}")


def cli_app():
    app = typer.Typer()
    app.command()(run_single_worker)
    app.command()(run_pool)
    app.command()(run_ray)
    app.command()(run_dask)
    app()

if __name__ == "__main__":
    cli_app()
