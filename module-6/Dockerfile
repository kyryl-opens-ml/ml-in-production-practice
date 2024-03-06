 FROM huggingface/transformers-pytorch-gpu:4.22.1 as base

WORKDIR /app

ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8

RUN pip install pip --upgrade 
COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt

RUN pip install kserve==0.10.1 ray==2.0.0


RUN ln -s /usr/bin/python3 /usr/bin/python

ENV PYTHONPATH /app
COPY . . 

CMD [ "bash" ]

FROM base AS app-fastapi
CMD uvicorn --host 0.0.0.0 --port 8080 --workers 4 serving.fast_api:app 

FROM base AS app-kserve
ENTRYPOINT ["python", "serving/kserve_api.py"]


FROM base AS app-kserve-transformer

ENTRYPOINT ["python", "serving/kserve_transformer.py"]

