import sys
import modal

app = modal.App("ml-in-production-module-1")

@app.function()
def f(i):
    if i % 2 == 0:
        print("hello", i)
    else:
        print("world", i, file=sys.stderr)

    return i * i

@app.local_entrypoint()
def main():

    # run the function remotely on Modal
    print(f.remote(1000))

    # run the function in parallel and remotely on Modal
    total = 0
    for ret in f.map(range(20)):
        total += ret

    print(total)
