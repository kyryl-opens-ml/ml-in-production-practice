import argilla as rg
from datasets import load_dataset
import sqlite3
import json
from typing import Dict
from retry import retry
from tqdm import tqdm
client = rg.Argilla(api_url="http://0.0.0.0:6900", api_key="admin.apikey")


from openai import OpenAI

def get_sqllite_schema(db_name: str) -> str:
    with sqlite3.connect(db_name) as conn:
        cursor = conn.cursor()
        
        cursor.execute("SELECT 'CREATE TABLE ' || name || ' (' || sql || ');' FROM sqlite_master WHERE type='table';")
        db_schema_records = cursor.fetchall()
        
        db_schema = [x[0] for x in db_schema_records]
        db_schema = "\n".join(db_schema)
    
    return db_schema


@retry(tries=3, delay=1)
def generate_synthetic_example(db_schema: str) -> Dict[str, str]:
    client = OpenAI()
    
    prompt = f"""
    Corresponding database schema: {db_schema}
    
    Please generate a example of what user might ask from this datasebase: in plan text and in SQL.
    Return only JSON with next format {{"user_text": '...', "sql": "...."}}
    """

    chat_completion = client.chat.completions.create(
        messages=[
        {
            "role": "system",
            "content": "You are SQLite and SQL expert.",
        },            
            {
                "role": "user",
                "content": prompt,
            }
        ],
        model="gpt-4o",
        response_format={ "type": "json_object" },
        temperature=1
    )
    sample = json.loads(chat_completion.choices[0].message.content)
    assert "user_text" in sample
    assert "sql" in sample
    return sample

def create_text2sql_dataset_synthetic(num_samples: int = 10):
    db_schema = get_sqllite_schema('examples/chinook.db')
    samples = []
    for _ in tqdm(range(num_samples)):
        sample = generate_synthetic_example(db_schema=db_schema)
        samples.append(sample)

    guidelines = f"""
    Please examine the given SQL question and context. Write the correct SQL query that accurately answers the question based on the context provided. Ensure the query follows SQL syntax and logic correctly.

    DB schema \n\n{db_schema}\n\n

    To verify the query:

	- Download the database file here: https://www.sqlitetutorial.net/wp-content/uploads/2018/03/chinook.zip
	- Install SQLite.
	- Run sqlite3 chinook.db.
    """

    dataset_name = "text2sql-chinook-synthetic"
    settings = rg.Settings(
        guidelines=guidelines,
        fields=[
            rg.TextField(
                name="schema",
                title="Schema",
                use_markdown=True,
            ),
            rg.TextField(
                name="sync_query",
                title="Query",
                use_markdown=False,
            ),            
            rg.TextField(
                name="sync_sql",
                title="SQL",
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
        workspace="admin",
        settings=settings,
        client=client,
    )
    dataset.create()

    records = []
    for sample in samples:
        x = rg.Record(
            fields={
                "sync_sql": sample['sql'],
                "sync_query": sample['user_text'],
                "schema": db_schema,
            },
        )
        records.append(x)
    dataset.records.log(records, batch_size=1000)


if __name__ == "__main__":
    create_text2sql_dataset_synthetic()
