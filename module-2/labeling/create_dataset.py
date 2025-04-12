from datasets import load_dataset

import argilla as rg

client = rg.Argilla(api_url="http://0.0.0.0:6900", api_key="argilla.apikey")
WORKSPACE_NAME = "admin"

def create_workspace():
    workspaces = client.workspaces
    if WORKSPACE_NAME not in workspaces:
        workspace = rg.Workspace(name=WORKSPACE_NAME, client=client)
        workspace.create()



def create_text2sql_dataset():
    create_workspace()
    
    guidelines = """
    Please examine the given SQL question and context. Write the correct SQL query that accurately answers the question based on the context provided. Ensure the query follows SQL syntax and logic correctly.
    """

    dataset_name = "text2sql"

    settings = rg.Settings(
        guidelines=guidelines,
        fields=[
            rg.TextField(
                name="query",
                title="Query",
                use_markdown=False,
            ),
            rg.TextField(
                name="schema",
                title="Schema",
                use_markdown=True,
            ),
        ],
        questions=[
            rg.TextQuestion(
                name="sql",
                title="Please write SQL for this query",
                description="Please write SQL for this query",
                required=True,
                use_markdown=True,
            )
        ],
    )
    dataset = rg.Dataset(
        name=dataset_name,
        settings=settings,
        workspace=WORKSPACE_NAME,
        client=client,
    )
    dataset.create()

    dataset = client.datasets(name=dataset_name)
    data = load_dataset("b-mc2/sql-create-context")
    records = []
    for idx in range(len(data["train"])):
        x = rg.Record(
            fields={
                "query": data["train"][idx]["question"],
                "schema": data["train"][idx]["context"],
            },
        )
        records.append(x)
    dataset.records.log(records, batch_size=1000)


if __name__ == "__main__":
    create_text2sql_dataset()
