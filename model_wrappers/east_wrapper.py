import cv2
import math
import time
import os

from time import time


class EASTWrapper:

    def __init__(self, model_weights):
        self.model_weights = model_weights
        self.conf_threshold = 0.7
        self.nms_threshold = 0.7
        self.new_width = 512
        self.new_height = 512
        self.load_model()

    def load_model(self):
        self.net = cv2.dnn.readNet(self.model_weights)
        self.layer_names = [
                 "feature_fusion/Conv_7/Sigmoid",
                 "feature_fusion/concat_3"
                 ]

    def detect(self, image):
        start = time()
        image_to_draw = image.copy()
        h, w = image_to_draw.shape[:2]

        ratio_width = w / float(self.new_width)
        ratio_height = h / float(self.new_height)

        blob = cv2.dnn.blobFromImage(image_to_draw, 1.0, (self.new_width, self.new_height), (123.68, 116.78, 103.94), True, False)

        # Run the model
        self.net.setInput(blob)
        outs = self.net.forward(self.layer_names)

        # Get scores and geometry
        scores = outs[0]
        geometry = outs[1]
        [boxes, confidences] = self.decode(scores, geometry, self.conf_threshold)

        indices = cv2.dnn.NMSBoxesRotated(boxes, confidences, self.conf_threshold, self.nms_threshold)
        text_parts = []
        for i in indices:
            # get 4 corners of the rotated rect
            vertices = cv2.boxPoints(boxes[i[0]])
            # scale the bounding box coordinates based on the respective ratios
            for j in range(4):
                vertices[j][0] *= ratio_width
                vertices[j][1] *= ratio_height
            points = []
            for j in range(4):
                p1 = (vertices[j][0], vertices[j][1])
                p2 = (vertices[(j + 1) % 4][0], vertices[(j + 1) % 4][1])
                points.append((p1, p2))
                cv2.line(image_to_draw, p1, p2, (0, 255, 0), 1)
            x = [int(x) for point in points for y, x in point]
            y = [int(y) for point in points for y, x in point]
            min_x = max(0, min(x) - 5)
            min_y = max(0, min(y) - 5)
            max_x = min(w, max(x) + 5)
            max_y = max(h, max(y) + 5)
            text_parts.append(image[min_x:max_x, min_y:max_y, :])
        return image_to_draw, text_parts, time() - start

    def decode(self, scores, geometry, scoreThresh):
        detections = []
        confidences = []

        height = scores.shape[2]
        width = scores.shape[3]
        for y in range(0, height):

            # Extract data from scores
            scoresData = scores[0][0][y]
            x0_data = geometry[0][0][y]
            x1_data = geometry[0][1][y]
            x2_data = geometry[0][2][y]
            x3_data = geometry[0][3][y]
            anglesData = geometry[0][4][y]
            for x in range(0, width):
                score = scoresData[x]

                # If score is lower than threshold score, move to next x
                if (score < scoreThresh):
                    continue

                # Calculate offset
                offsetX = x * 4.0
                offsetY = y * 4.0
                angle = anglesData[x]

                # Calculate cos and sin of angle
                cosA = math.cos(angle)
                sinA = math.sin(angle)
                h = x0_data[x] + x2_data[x]
                w = x1_data[x] + x3_data[x]

                # Calculate offset
                offset = (
                [offsetX + cosA * x1_data[x] + sinA * x2_data[x], offsetY - sinA * x1_data[x] + cosA * x2_data[x]])

                # Find points for rectangle
                p1 = (-sinA * h + offset[0], -cosA * h + offset[1])
                p3 = (-cosA * w + offset[0], sinA * w + offset[1])
                center = (0.5 * (p1[0] + p3[0]), 0.5 * (p1[1] + p3[1]))
                detections.append((center, (w, h), -1 * angle * 180.0 / math.pi))
                confidences.append(float(score))

        # Return detections and confidences
        return [detections, confidences]


if __name__ == '__main__':
    image = cv2.imread(os.path.join("data", "images_to_test", "test_detector.jpg"))
    east = EASTWrapper(os.path.join("data", "east-model", "frozen_east_text_detection.pb"))
    image, text_parts, _ = east.detect(image)
    cv2.imshow("Detected", image)
    key = cv2.waitKey(0)
    if key == 27:
        pass
