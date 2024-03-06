# FROM huggingface/transformers-pytorch-gpu:4.22.1 as base
FROM huggingface/transformers-pytorch-gpu:4.35.2 as base

WORKDIR /app

ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8

RUN pip install pip --upgrade 
COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt


RUN ln -s /usr/bin/python3 /usr/bin/python

ENV PYTHONPATH /app
COPY . . 

CMD [ "bash" ]

# Streamlit UI docker image
FROM base AS app-streamlit
CMD streamlit run --server.address 0.0.0.0 --server.port 8080 serving/ui_app.py


# Fast API docker image
FROM base AS app-fastapi
CMD uvicorn --host 0.0.0.0 --port 8080 --workers 4 serving.fast_api:app 


FROM base AS app-seldon

# Port for GRPC
EXPOSE 5000
# Port for REST
EXPOSE 9000

# Define environment variables
ENV MODEL_NAME SeldonAPI
ENV SERVICE_TYPE MODEL
# COPY /app/serving/seldon_api.py /app/SeldonAPI.py
COPY serving/seldon_api.py /app/SeldonAPI.py

RUN chown -R 8888 /app
RUN mkdir /.cache
RUN chmod 777 /.cache
RUN mkdir /.config
RUN chmod 777 /.config

CMD exec seldon-core-microservice $MODEL_NAME --service-type $SERVICE_TYPE


FROM base AS app-kserve
ENV WANDB_API_KEY=cb86168a2e8db7edb905da69307450f5e7867d66
RUN pip install protobuf==3.20.*
ENTRYPOINT ["python", "serving/kserve_api.py"]
