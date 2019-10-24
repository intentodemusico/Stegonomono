#FLASH
from flask import Flask

app = Flask(__name__)

@app.route("/API/prueba")
def hello():
    return "Prueba de API, NO USAR COMO FINAL"
