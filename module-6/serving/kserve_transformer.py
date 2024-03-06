import argparse
import datetime
import json
import logging
import time
from typing import Dict, Union

import boto3
import kserve
from cloudevents.http import CloudEvent
from kserve import InferRequest, InferResponse
from kserve.protocol.grpc.grpc_predict_v2_pb2 import ModelInferResponse

logging.basicConfig(level=kserve.constants.KSERVE_LOGLEVEL)

session = boto3.Session()
client = session.client("s3", endpoint_url="http://minio-service:9000", aws_access_key_id="minio", aws_secret_access_key="minio123")
digits_bucket = "output"


class ImageTransformer(kserve.Model):
    def __init__(self, name: str, predictor_host: str):
        super().__init__(name)
        self.predictor_host = predictor_host
        self._key = None

    def preprocess(self, inputs: Union[Dict, CloudEvent, InferRequest], headers: Dict[str, str] = None) -> Union[Dict, InferRequest]:

        logging.info("Received inputs %s", inputs)
        data = json.loads(inputs.get_data().decode("utf-8"))
        inputs = data
        if inputs["EventName"] == "s3:ObjectCreated:Put":
            bucket = inputs["Records"][0]["s3"]["bucket"]["name"]
            key = inputs["Records"][0]["s3"]["object"]["key"]
            self._key = key
            client.download_file(bucket, key, "/tmp/" + key)

            with open("/tmp/" + key, "r") as f:
                instances = json.load(f)["instances"]
                logging.info(f"instances {instances}")

            return {"instances": instances}
        raise Exception("unknown event")

    def postprocess(self, response: Union[Dict, InferResponse, ModelInferResponse], headers: Dict[str, str] = None) -> Union[Dict, ModelInferResponse]:
        logging.info(
            f"response: {response}",
        )
        predictions = response["predictions"]
        logging.info(f"predictions: {predictions}")

        upload_path = f"predictions_{time.time()}-{self._key}"
        with open(upload_path, "w") as f:
            json.dump(predictions, f)

        client.upload_file(upload_path, digits_bucket, upload_path)
        logging.info(f"Image {self._key} successfully uploaded to {upload_path}")
        return response


DEFAULT_MODEL_NAME = "custom-model"

parser = argparse.ArgumentParser(parents=[kserve.model_server.parser])
parser.add_argument("--model_name", default=DEFAULT_MODEL_NAME, help="The name that the model is served under.")
parser.add_argument("--predictor_host", help="The URL for the model predict function", required=True)

args, _ = parser.parse_known_args()

if __name__ == "__main__":
    transformer = ImageTransformer(args.model_name, predictor_host=args.predictor_host)
    server = kserve.ModelServer()
    server.start(models=[transformer])
