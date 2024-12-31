import os
from typing import Union

from fastapi import FastAPI

app = FastAPI()

files = os.listdir("texts")


@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.get("/texts/{n}")
def texts(n: int):
    if n > len(files):
        return f"n too large, max {len(files)}"

    out = []
    for i in range(n):
        with open(os.path.join("texts", files[i])) as file:
            out.append(file.readlines())
    return out


@app.get("/fibo/{n}")
def fibo(n: int):
    return fiboRec(n)


def fiboRec(n: int):
    if n <= 2:
        return 1
    return fiboRec(n - 1) + fiboRec(n - 2)
