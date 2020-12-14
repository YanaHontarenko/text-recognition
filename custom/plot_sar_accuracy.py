import sys
import os
sys.path.append(os.path.join(os.getcwd(), "sar-pytorch"))
import numpy as np

from dataset.dataset import iiit5k_mat_extractor

with open(os.path.join("data", "sar-results", "predict.txt"), "r") as file:
    data = file.read()
    data = np.array([line.split(" ") for line in data.split("\n")])

annotation = iiit5k_mat_extractor(os.path.join("data", "IIIT5K", "testdata.mat"))
ann_dict = {}
for ann in annotation:
    ann_dict[ann[0][5:]] = ann[1]

count = 0
for image, predict in data:
    if ann_dict[image] == predict:
        count += 1

print(f"Full sequence accuracy: {count/len(data)}")