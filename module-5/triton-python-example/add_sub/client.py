import sys

import numpy as np
import tritonclient.http as httpclient
from tritonclient.utils import *

model_name = "add_sub"
shape = [4]

with httpclient.InferenceServerClient("localhost:8000") as client:

    input0_data = np.random.rand(*shape).astype(np.float32)
    input1_data = np.random.rand(*shape).astype(np.float32)


    inputs = [
        httpclient.InferInput("INPUT0", input0_data.shape, np_to_triton_dtype(input0_data.dtype)),
        httpclient.InferInput("INPUT1", input1_data.shape, np_to_triton_dtype(input1_data.dtype)),
    ]

    inputs[0].set_data_from_numpy(input0_data)
    inputs[1].set_data_from_numpy(input1_data)

    outputs = [
        httpclient.InferRequestedOutput("OUTPUT0"),
        httpclient.InferRequestedOutput("OUTPUT1"),
    ]

    response = client.infer(model_name, inputs, request_id=str(1), outputs=outputs)

    result = response.get_response()
    output0_data = response.as_numpy("OUTPUT0")
    output1_data = response.as_numpy("OUTPUT1")

    print(
        "INPUT0 ({}) + INPUT1 ({}) = OUTPUT0 ({})".format(
            input0_data, input1_data, output0_data
        )
    )
    print(
        "INPUT0 ({}) - INPUT1 ({}) = OUTPUT1 ({})".format(
            input0_data, input1_data, output1_data
        )
    )

    if not np.allclose(input0_data + input1_data, output0_data):
        print("add_sub example error: incorrect sum")
        sys.exit(1)

    if not np.allclose(input0_data - input1_data, output1_data):
        print("add_sub example error: incorrect difference")
        sys.exit(1)

    print("PASS: add_sub")
    sys.exit(0)