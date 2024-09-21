from locust import HttpUser, TaskSet, task, between
from datasets import load_dataset
import random
import json

class LoadTestUser(HttpUser):
    wait_time = between(1, 5)

    def on_start(self):
        self.dataset = load_dataset("gretelai/synthetic_text_to_sql", split='train')
        self.dataset_size = len(self.dataset)

    @task
    def generate_sql(self):
        index = random.randint(0, self.dataset_size - 1)
        sample = self.dataset[index]

        sql_context = sample.get('sql_context', 'No context provided')
        sql_prompt = sample.get('sql_prompt', 'No prompt provided')

        input_text = f"Generate sql for this context: {sql_context} for this query: {sql_prompt}"

        payload = {
            "inputs": input_text
        }

        headers = {
            "accept": "application/json",
            "Content-Type": "application/json"
        }

        with self.client.post("/generate", 
                              data=json.dumps(payload), 
                              headers=headers, 
                              name="/generate", 
                              catch_response=True) as response:
            if response.status_code != 200:
                response.failure(f"Failed with status code {response.status_code}")
            else:
                response.success()
