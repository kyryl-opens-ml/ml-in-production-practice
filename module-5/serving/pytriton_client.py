import numpy as np
from pytriton.client import ModelClient


# https://triton-inference-server.github.io/pytriton/latest/clients/
def main():
    text = np.array(
        [
            ["one day I will see the world"],
        ]
    )
    text = np.char.encode(text, "utf-8")

    with ModelClient("0.0.0.0", "predictor_a") as client:
        result_dict = client.infer_batch(text=text)
        print(result_dict["probs"])


if __name__ == "__main__":
    main()
