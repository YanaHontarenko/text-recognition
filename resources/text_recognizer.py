from flask_restful import Resource, reqparse
from werkzeug.datastructures import FileStorage


class Recognizer(Resource):
    _parser = reqparse.RequestParser()
    _parser.add_argument('image', type=FileStorage, required=True, help="This field cannot be blank.", location='files')

    def post(self, model_type):
        return "Returned result"
