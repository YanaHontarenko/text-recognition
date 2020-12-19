import os

from flask import render_template, make_response
from flask_restful import Resource, reqparse
from models.text_recognizer import TextRecognizerModel
from werkzeug.datastructures import FileStorage


class Recognizer(Resource):
    _parser = reqparse.RequestParser()
    _parser.add_argument('image', type=FileStorage, required=True, help="This field cannot be blank.", location='files')

    def post(self, model_type):
        data = self._parser.parse_args()
        image = data['image'].read()
        common_time = 0
        if model_type == "craft-crnn":
            image, common_time = TextRecognizerModel.craft_crnn(image)
        elif model_type == "craft-sar":
            image, common_time = TextRecognizerModel.craft_sar(image)
        elif model_type == "east-crnn":
            image, common_time = TextRecognizerModel.east_crnn(image)
        elif model_type == "east-sar":
            image, common_time = TextRecognizerModel.east_sar(image)
        else:
            pass
            # TODO: Add masktextspotter model
        with open(os.path.join("static", "tempImg.jpg"), "wb") as file:
            file.write(image)
        with open(os.path.join("static", "time.txt"), "w") as file:
            file.write(str(common_time))
        return self.redirect(model_type)

    def redirect(self, model_type):
        return "/recognize/" + model_type

    def get(self, model_type):
        with open(os.path.join("static", "time.txt"), "w") as file:
            common_time = float(file.read())
        result_string = TextRecognizerModel.create_result_string(model_type, common_time)
        return make_response(render_template("recognized.html", message=result_string, image_path="tempImg.jpg"))