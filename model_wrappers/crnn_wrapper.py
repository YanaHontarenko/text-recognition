import sys
import os
sys.path.append(os.path.join(os.getcwd(), "CRNN_Tensorflow"))

import cv2
import numpy as np
import tensorflow as tf

from config import global_config
from crnn_model import crnn_net
from data_provider import tf_io_pipline_fast_tools
from time import time

CFG = global_config.cfg


class CRNNWrapper():

    def __init__(self, weights_path, char_dict_path, ord_map_dict_path):
        self.weights_path = weights_path
        self.char_dict_path = char_dict_path
        self.ord_map_dict_path = ord_map_dict_path
        self.load_model()

    def load_model(self):
        self.inputdata = tf.placeholder(
            dtype=tf.float32,
            shape=[1, CFG.ARCH.INPUT_SIZE[1], CFG.ARCH.INPUT_SIZE[0], CFG.ARCH.INPUT_CHANNELS],
            name='input'
        )

        self.codec = tf_io_pipline_fast_tools.CrnnFeatureReader(
            char_dict_path=self.char_dict_path,
            ord_map_dict_path=self.ord_map_dict_path
        )

        net = crnn_net.ShadowNet(
            phase='test',
            hidden_nums=CFG.ARCH.HIDDEN_UNITS,
            layers_nums=CFG.ARCH.HIDDEN_LAYERS,
            num_classes=CFG.ARCH.NUM_CLASSES
        )

        inference_ret = net.inference(
            inputdata=self.inputdata,
            name='shadow_net',
            reuse=False
        )

        self.decodes, _ = tf.nn.ctc_greedy_decoder(
            inference_ret,
            CFG.ARCH.SEQ_LENGTH * np.ones(1),
            merge_repeated=True
        )

        self.saver = tf.train.Saver()

        # config tf session
        self.sess_config = tf.ConfigProto(allow_soft_placement=True)
        self.sess_config.gpu_options.per_process_gpu_memory_fraction = CFG.TEST.GPU_MEMORY_FRACTION
        self.sess_config.gpu_options.allow_growth = CFG.TEST.TF_ALLOW_GROWTH

    def recognize(self, image):
        image = cv2.resize(image, dsize=tuple(CFG.ARCH.INPUT_SIZE), interpolation=cv2.INTER_LINEAR)
        image = np.array(image, np.float32) / 127.5 - 1.0

        start = time()
        sess = tf.Session(config=self.sess_config)

        with sess.as_default():
            self.saver.restore(sess=sess, save_path=self.weights_path)
            preds = sess.run(self.decodes, feed_dict={self.inputdata: [image]})
            preds = self.codec.sparse_tensor_to_str(preds[0])[0]

        sess.close()
        return preds, time() - start


if __name__ == '__main__':
    crnn = CRNNWrapper(os.path.join("data", "crnn-pretrained", "shadownet.ckpt-80000"),
                        os.path.join("data", "crnn-dicts", "char_dict_en.json"),
                        os.path.join("data", "crnn-dicts", "ord_map_en.json"))
    image = cv2.imread(os.path.join("data", "images_to_test", "test_recognizer.png"))
    text, t = crnn.recognize(image)
    cv2.imshow(text, image)
    key = cv2.waitKey(0)
    if key == 27:
        pass
