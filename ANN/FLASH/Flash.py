# -*- coding: utf-8 -*-
"""
Created on Tue Dec 17 14:10:02 2019

@author: INTENTODEMUSICO
"""
#!/usr/bin/python
from flask import request, url_for
from flask_api import FlaskAPI, status, exceptions

app = FlaskAPI(__name__)
@app.route("/")
def index():
    return "/API/image/ para subir mediante un post -> el resultado se entregará en la cabecera de respuesta"

@app.route("/API/image/", methods=['GET', 'POST', 'PUT'])
def result():
    if request.method == 'GET':
        return "Perro, debe usar POST"
    if request.method == "POST":
        test = str(request.data.get('test', ''))
        print("aaa",test)
        test+="El angarita resultado"
        return (test,"200")
    return("Holi")
app.run(host="0.0.0.0", port=2012)