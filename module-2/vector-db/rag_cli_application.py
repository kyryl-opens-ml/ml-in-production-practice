import typer
import random
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
import lance
import numpy as np
import os
import json
import lancedb

app = typer.Typer()

@app.command()
def create_new_vector_db(table_name: str = "my-rag-app", number_of_documents: int = 1000):
    dataset = load_dataset("b-mc2/sql-create-context")
    model = SentenceTransformer('all-MiniLM-L6-v2')

    docs = random.sample(list(dataset['train']), k=number_of_documents)

    texts = [json.dumps(doc) for doc in docs]
    embeddings = model.encode(texts)

    data = [{
        'id': idx, 
        'text': texts[idx], 
        'vector': embeddings[idx], 
        'answer': docs[idx]['answer'],
        'question': docs[idx]['question'],
        'context': docs[idx]['context']

        } for idx in range(len(texts))]
    
    uri = ".lancedb"
    db = lancedb.connect(uri)
    lance_table = db.create_table(table_name, data=data)
    lance_table.create_index()

    typer.echo(f"Lance table {table_name} created with {number_of_documents} documents.")

@app.command()
def query_existing_vector_db(query: str = "test", table_name: str = "my-rag-app", top_n: int = 2):
    model = SentenceTransformer('all-MiniLM-L6-v2')
    query_embedding = model.encode(query)

    uri = ".lancedb"
    db = lancedb.connect(uri)
    tbl = db.open_table(table_name)

    results = tbl.search(query_embedding).limit(top_n).to_list()
    typer.echo("Search result:")
    for result in results:
        typer.echo("RESULT")
        typer.echo(result['answer'])
        typer.echo(result['context'])
        typer.echo(result['question'])


if __name__ == "__main__":
    app()
