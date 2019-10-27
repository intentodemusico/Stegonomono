#FLASH
from flask import Flask

app = Flask(__name__)

@app.route("/API")
def hello():
    return "Prueba de API, NO USAR COMO FINAL"
