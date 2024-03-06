
import typer
import numpy as np
import pandas as pd
from tqdm import tqdm



from pymilvus import connections
from pymilvus import CollectionSchema, FieldSchema, DataType
from pymilvus import Collection


def load_data(collection_name: str, collection_size: int, vec_dim: int):

    doc_id = FieldSchema(
        name="doc_id", 
        dtype=DataType.INT64, 
        is_primary=True, 
    )
    doc_vector = FieldSchema(
        name="doc_vector", 
        dtype=DataType.FLOAT_VECTOR, 
        dim=vec_dim
    )
    schema = CollectionSchema(
        fields=[doc_id, doc_vector], 
        description="Test"
    )
    collection = Collection(
        name=collection_name, 
        schema=schema, 
        using='default', 
        )
    
    batch_size = 1000
    for idx in tqdm(range(0, collection_size, batch_size)):
        embeddings_batch = np.random.rand(batch_size * vec_dim).reshape((batch_size, vec_dim))
        data = [np.arange(idx, idx + embeddings_batch.shape[0]), embeddings_batch]
        mr = collection.insert(data)
        print(mr)


def build_index(collection_name: str):
    index_params = {
        "metric_type":"L2",
        "index_type":"IVF_SQ8",
         "params":{"nlist":1024}
    }
    collection = Collection(collection_name)
    res = collection.create_index(
        field_name="doc_vector", 
        index_params=index_params
    )
    print(res)

def data_ingest(collection_name: str = 'test', collection_size: int = 100_000, vec_dim: int = 768, host: str = 'localhost', port: str = '19530'):
    print("connecting ...")
    connections.connect(
        alias="default", 
        host=host, 
        port=port
    )
    print("loading data")
    load_data(collection_name=collection_name, collection_size=collection_size, vec_dim=vec_dim)


    print("building index")
    build_index(collection_name=collection_name)


def search_index(collection_name: str = 'test', vec_dim: int = 768, host: str = 'localhost', port: str = '19530'):
    connections.connect(
        alias="default", 
        host=host, 
        port=port
    )    

    collection = Collection(collection_name)
    collection.load()

    q_vecto = np.random.rand(vec_dim)

    search_params = {"metric_type": "L2", "params": {"nprobe": 10}, "offset": 5}
    results = collection.search(
        data=[q_vecto], 
        anns_field="doc_vector", 
        param=search_params,
        limit=25, 
        expr=None,
        consistency_level="Strong"
    )   
    res = results[0].ids 

    print(f"results {res}")

if __name__ == "__main__":
    app = typer.Typer()
    app.command()(search_index)
    app.command()(data_ingest)
    app()
