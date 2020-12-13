import os

from flask import Flask, render_template
from flask_restful import Api

from resources.text_recognizer import Recognizer

app = Flask(__name__)
api = Api(app)


@app.after_request
def add_header(r):
    """
    Add headers to both force latest IE rendering engine or Chrome Frame,
    and also to cache the rendered page for 10 minutes.
    """
    r.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    r.headers["Pragma"] = "no-cache"
    r.headers["Expires"] = "0"
    r.headers['Cache-Control'] = 'public, max-age=0'
    return r


@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")

api.add_resource(Recognizer, "/recognize/<string:model_type>")
if __name__ == '__main__':
    PORT = int(os.getenv("PORT", 8080))
    HOST = os.getenv("HOST", "localhost")
    app.run(port=PORT, host=HOST)
