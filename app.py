import os

from flask import Flask, render_template
from flask_restful import Api

from resources.text_recognizer import Recognizer

app = Flask(__name__)
api = Api(app)


@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")

api.add_resource(Recognizer, "/recognize/<string:model_type>")
if __name__ == '__main__':
    PORT = int(os.getenv("PORT", 8080))
    HOST = os.getenv("HOST", "localhost")
    app.run(port=PORT, host=HOST)
