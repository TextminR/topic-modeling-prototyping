from flask import Flask
import os

app = Flask(__name__)


files = os.listdir("texts")

@app.route("/")
def hello_world():
    return "<p>Hello, World!</p>"

@app.route("/texts/<amount>")
def texts(amount: int):
    if int(amount) > len(files):
        return f"n too large, max {len(files)}"

    out = []
    for i in range(int(amount)):
        with open(os.path.join("texts", files[i])) as file:
            out.append(file.readlines())
    return out


@app.route("/fibo/<n>")
def fibo(n):
    n = int(n)
    return str(fiboRec(n))


def fiboRec(n: int):
    if n <= 2:
        return 1
    return fiboRec(n - 1) + fiboRec(n - 2)
