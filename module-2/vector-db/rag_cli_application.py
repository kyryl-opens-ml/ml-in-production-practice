import random

import lancedb
import typer
from datasets import load_dataset
from sentence_transformers import SentenceTransformer

app = typer.Typer()
MODEL_NAME = "paraphrase-MiniLM-L3-v2"


@app.command()
def create_new_vector_db(
    table_name: str = "my-rag-app", number_of_documents: int = 1000, uri=".lancedb"
):
    dataset = load_dataset("b-mc2/sql-create-context")
    model = SentenceTransformer(MODEL_NAME)

    docs = random.sample(list(dataset["train"]), k=number_of_documents)

    texts = [doc["question"] for doc in docs]
    embeddings = model.encode(texts, show_progress_bar=True)

    data = [
        {
            "id": idx,
            "text": texts[idx],
            "vector": embeddings[idx],
            "answer": docs[idx]["answer"],
            "question": docs[idx]["question"],
            "context": docs[idx]["context"],
        }
        for idx in range(len(texts))
    ]

    db = lancedb.connect(uri)
    lance_table = db.create_table(table_name, data=data)
    lance_table.create_index()

    typer.echo(
        f"Lance table {table_name} created with {number_of_documents} documents."
    )


@app.command()
def query_existing_vector_db(
    query: str = "What was ARR last year?",
    table_name: str = "my-rag-app",
    top_n: int = 1,
    uri=".lancedb",
):
    model = SentenceTransformer(MODEL_NAME)
    query_embedding = model.encode(query)

    db = lancedb.connect(uri)
    tbl = db.open_table(table_name)

    results = tbl.search(query_embedding).limit(top_n).to_list()
    typer.echo("Search result:")
    for result in results:
        typer.echo("RESULT")
        typer.echo(result["answer"])
        typer.echo(result["context"])
        typer.echo(result["question"])


if __name__ == "__main__":
    app()
