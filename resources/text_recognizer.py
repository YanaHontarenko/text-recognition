import os

from flask import render_template, url_for, redirect, make_response
from flask_restful import Resource, reqparse
from werkzeug.datastructures import FileStorage


class Recognizer(Resource):
    _parser = reqparse.RequestParser()
    _parser.add_argument('image', type=FileStorage, required=True, help="This field cannot be blank.", location='files')

    def post(self, model_type):
        data = self._parser.parse_args()
        image = data['image'].read()
        with open(os.path.join("static", "tempImg.jpg"), "wb") as file:
            file.write(image)
        return self.redirect(model_type)

    def redirect(self, model_type):
        return "/recognize/" + model_type

    def get(self, model_type):
        return make_response(render_template("recognized.html", message="Used time", image_path="tempImg.jpg"))