from fastapi import FastAPI
from pydantic import BaseModel
import typer
import uvicorn
import boto3
import json
from botocore.exceptions import ClientError
import time
from tinydb import TinyDB, Query
from uuid import uuid4

AWS_ENDPOINT = "http://localhost:3001"
QUEUE_NAME = "ml-infernece"
QUEUE_URL = f"{AWS_ENDPOINT}/1/{QUEUE_NAME}"

class RecordManager:
    def __init__(self, db_path='db.json'):
        self.db = TinyDB(db_path)
        self.query = Query()
    
    def add_record(self):
        record_id = str(uuid4())
        self.db.insert({'id': record_id, 'status': 'pending'})
        return record_id
    
    def update_record(self, record_id, result_data):
        update_data = {'status': 'done'}
        update_data.update(result_data)
        self.db.update(update_data, self.query.id == record_id)
    
    def pull_record(self, record_id):
        return self.db.get(self.query.id == record_id)
    

web_app = FastAPI()
record_manager = RecordManager()



def get_or_create_queue(queue_name, aws_region='us-east-1'):
    sqs = boto3.client('sqs', region_name=aws_region, endpoint_url=AWS_ENDPOINT)
    try:
        response = sqs.get_queue_url(QueueName=queue_name)
        return response['QueueUrl']
    except ClientError as e:
        if e.response['Error']['Code'] == 'QueueDoesNotExist':
            response = sqs.create_queue(QueueName=queue_name)
            return response['QueueUrl']
        else:
            raise

def submit_message_to_sqs(queue_url, data, aws_region='us-east-1'):
    sqs = boto3.client('sqs', region_name=aws_region, endpoint_url=AWS_ENDPOINT)
    message_body = json.dumps(data)
    try:
        response = sqs.send_message(QueueUrl=queue_url, MessageBody=message_body)
        return response
    except ClientError as e:
        raise e
    
def submit_job(data):
    call_id = record_manager.add_record()
    submit_message_to_sqs(queue_url=QUEUE_URL, data={"data": data, "call_id": call_id}, aws_region='us-east-1')
    return call_id

def get_job_result(call_id):
    record = record_manager.pull_record(record_id=call_id)
    return record

class SubmitJobRequest(BaseModel):
    data: str

@web_app.post("/submit_job")
async def submit_job_endpoint(request: SubmitJobRequest):
    call_id = submit_job(request.data)
    return {"call_id": call_id}

@web_app.get("/get_job_result")
async def get_job_result_endpoint(call_id: str):
    result = get_job_result(call_id)
    return result


def run_api():
    get_or_create_queue(queue_name=QUEUE_NAME)
    uvicorn.run(web_app, host="0.0.0.0", port=8000)

def receive_messages_from_sqs(queue_url, max_number=1, aws_region='us-east-1'):
    sqs = boto3.client('sqs', region_name=aws_region, endpoint_url=AWS_ENDPOINT)
    try:
        response = sqs.receive_message(
            QueueUrl=queue_url,
            MaxNumberOfMessages=max_number,
            WaitTimeSeconds=10
        )
        messages = response.get('Messages', [])
        for message in messages:
            sqs.delete_message(
                QueueUrl=queue_url,
                ReceiptHandle=message['ReceiptHandle']
            )
        return messages
    except ClientError as e:
        raise e

def process_job(data):
    time.sleep(60)
    return {"result": "some-result"}

def run_worker():
    while True:
        messages = receive_messages_from_sqs(queue_url=QUEUE_URL)
        print(f"pulled {messages}")
        if len(messages) == 0:
            time.sleep(1)
        for message in messages:
            data = json.loads(message['Body'])
            result_data = process_job(data=data)
            record_manager.update_record(record_id=data['call_id'], result_data=result_data)


def cli():
    app = typer.Typer()
    app.command()(run_api)
    app.command()(run_worker)
    app()

if __name__ == "__main__":
    cli()
