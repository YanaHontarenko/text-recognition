import sys
import os
sys.path.append(os.path.join(os.getcwd(), "sar-pytorch"))
import torch
import torch.nn.parallel
import torch.utils.data
import cv2
import numpy as np
# internal package
from dataset.dataset import dictionary_generator
from models.sar import sar
from utils.dataproc import end_cut
from time import time


class SARWrapper:

    def __init__(self, weights_path):
        self.weights_path = weights_path
        self.cuda = torch.cuda.is_available()
        self.height = 48
        self.width = 64
        self.feature_height = self.height // 4
        self.feature_width = self.width // 8
        self.channel = 3
        self.voc, self.char2id, self.id2char = dictionary_generator()
        self.output_classes = len(self.voc)
        self.embedding_dim = 512
        self.hidden_units = 512
        self.layers = 2
        self.keep_prob = 1.0
        self.seq_len = 40
        self.load_model()

    def load_model(self):
        if self.cuda:
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
        self.model = sar(self.channel, self.feature_height, self.feature_width, self.embedding_dim, self.output_classes,
                         self.hidden_units, self.layers, self.keep_prob, self.seq_len, self.device)
        if self.cuda:
            self.model.load_state_dict(torch.load(self.weights_path,
                                                  map_location=lambda storage, loc: storage),
                                       strict=False)
            self.model.cuda()
        else:
            self.model.load_state_dict(torch.load(self.weights_path,
                                                  map_location=lambda storage, loc: storage),
                                  strict=False)
            self.model = self.model.to(self.device)
        self.model = self.model.eval()

    def recognize(self, image):
        # image processing:
        start = time()
        image = cv2.resize(image, (self.width, self.height))  # resize
        image = (image - 127.5) / 127.5  # normalization to [-1,1]
        image = np.expand_dims(image, axis=0)
        image = torch.FloatTensor(image)  # convert to tensor [H, W, C]
        image = image.permute(0, 3, 1, 2)  # [C, H, W]
        if self.cuda:
            image = image.cuda()

        predict, att_weights, _, _ = self.model(image, 0)
        pred_choice = predict.max(2)[1][0]  # [batch_size, seq_len]
        predict = end_cut(pred_choice.detach().cpu().numpy(), self.char2id, self.id2char)
        return predict, time() - start


if __name__ == '__main__':
    sar = SARWrapper(os.path.join("data", "sar-models", "model_best_syn.pth"))
    image = cv2.imread(os.path.join("data", "images_to_test", "test_recognizer.png"))
    text, t = sar.recognize(image)
    cv2.imshow(text, image)
    key = cv2.waitKey(0)
    if key == 27:
        pass
