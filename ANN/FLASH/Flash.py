#!/usr/bin/python
from flask import Flask, escape, url_for

app = Flask(__name__)
@app.route("/")
def index():
    return "/API/image/upload para subir\n/API/image/result para el resultado"

@app.route("/API/image/result", methods=['GET', 'POST', 'PUT'])
def result():
    return "El número de Germán TEST"

@app.route("/API/image/upload", methods=['GET', 'POST', 'PUT'])
def upload():
    return "Suba su imagen pls"

app.run(host="0.0.0.0", port=2012)

