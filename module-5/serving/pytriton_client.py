import numpy as np
from pytriton.client import ModelClient

client = ModelClient("localhost", "predictor_a")
print(client.model_config)


sequence = np.array([
    ["one day I will see the world"],
])
sequence = np.char.encode(sequence, "utf-8")

result_dict = client.infer_batch(text=sequence)

data = np.array([1, 2, ], dtype=np.float32)
print(client.infer_sample(text="test"))


# kill -SIGINT 424
# Response like a list for an amazing engineers. Don’t add comments or overlap. Keep it concise.


