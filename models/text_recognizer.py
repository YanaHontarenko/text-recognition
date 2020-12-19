import cv2
import os

from model_wrappers.craft_wrapper import CRAFTWrapper
from model_wrappers.crnn_wrapper import CRNNWrapper
from model_wrappers.east_wrapper import EASTWrapper
from model_wrappers.sar_wrapper import SARWrapper


craft_model = CRAFTWrapper(os.path.join("data", "craft-model", "craft"))
crnn_model = CRNNWrapper(os.path.join("data", "crnn-model", "crnn"),
                         os.path.join("data", "crnn-dicts", "char_dict_en.json"),
                         os.path.join("data", "crnn-dicts", "ord_map_en.json"))
east_model = EASTWrapper(os.path.join("data", "east-model", "east"))
sar_model = SARWrapper(os.path.join("data", "sar-model", "sar"))
# TODO: Add masktextspotter model


class TextRecognizerModel:

    @staticmethod
    def craft_crnn(image):
        common_time = 0
        image, text_parts, time = craft_model.detect(image=image)
        common_time += time
        for x1, y1, x2, y2 in text_parts:
            text_part = image[x1:x2, y1:y2, :]
            text, time = crnn_model.recognize(text_part)
            cv2.putText(image, text, (x1, y1), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(0, 0, 255),
                        thickness=2)
            common_time += time
        return image, common_time

    @staticmethod
    def craft_sar(image):
        common_time = 0
        image, text_parts, time = craft_model.detect(image=image)
        common_time += time
        for x1, y1, x2, y2 in text_parts:
            text_part = image[x1:x2, y1:y2, :]
            text, time = sar_model.recognize(text_part)
            cv2.putText(image, text, (x1, y1), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(0, 0, 255), thickness=2)
            common_time += time
        return image, common_time

    @staticmethod
    def east_crnn(image):
        common_time = 0
        image, text_parts, time = east_model.detect(image=image)
        common_time += time
        for x1, y1, x2, y2 in text_parts:
            text_part = image[x1:x2, y1:y2, :]
            text, time = crnn_model.recognize(text_part)
            cv2.putText(image, text, (x1, y1), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(0, 0, 255), thickness=2)
            common_time += time
        return image, common_time

    @staticmethod
    def east_sar(image):
        common_time = 0
        image, text_parts, time = east_model.detect(image=image)
        common_time += time
        for x1, y1, x2, y2 in text_parts:
            text_part = image[x1:x2, y1:y2, :]
            text, time = sar_model.recognize(text_part)
            cv2.putText(image, text, (x1, y1), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(0, 0, 255), thickness=2)
            common_time += time
        return image, common_time

    # TODO: Add masktextspotter model

    @staticmethod
    def create_result_string(model_type, common_time):
        models = model_type.split("-")
        if len(models) == 2 and models[0] == "craft" and models[1] == "crnn":
            result_string = f"Prediction with CRAFT and CRNN models take {round(common_time, 2)} seconds"
        elif len(models) == 2 and models[0] == "craft" and models[1] == "sar":
            result_string = f"Prediction with CRAFT and SAR models take {round(common_time, 2)} seconds"
        elif len(models) == 2 and models[0] == "east" and models[1] == "crnn":
            result_string = f"Prediction with EAST and CRNN models take {round(common_time, 2)} seconds"
        elif len(models) == 2 and models[0] == "east" and models[1] == "sar":
            result_string = f"Prediction with EAST and SAR models take {round(common_time, 2)} seconds"
        else:
            result_string = f"Prediction with MaskTextSpotter model take {round(common_time, 2)} seconds"
        return result_string
