import logging
import numpy as np
from pytriton.client import ModelClient

# https://triton-inference-server.github.io/pytriton/latest/clients/
def main():
    sequence = np.array([
        ["one day I will see the world"],
    ])
    sequence = np.char.encode(sequence, "utf-8")

    with ModelClient("0.0.0.0", "BART") as client:
        result_dict = client.infer_batch(sequence)
        for output_name, output_data in result_dict.items():
            output_data = np.array2string(output_data, threshold=np.inf, max_line_width=np.inf, separator=",").replace("\n", "")
            print(f"{output_name}: {output_data}.")


if __name__ == "__main__":
    main()