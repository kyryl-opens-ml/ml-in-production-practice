from fastapi import FastAPI
from pydantic import BaseModel
import modal

app = modal.App("simple-queue")


@app.function()
def process_job(data):
    import time

    time.sleep(60)
    return {"result": data}


web_app = FastAPI()


def submit_job(data):
    process_job = modal.Function.lookup("simple-queue", "process_job")
    call = process_job.spawn(data)
    return call.object_id


def get_job_result(call_id):
    function_call = modal.functions.FunctionCall.from_id(call_id)
    try:
        result = function_call.get(timeout=0)
    except modal.exception.OutputExpiredError:
        result = {"result": "expired"}
    except TimeoutError:
        result = {"result": "pending"}
    return result


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


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(web_app, host="0.0.0.0", port=8000)
