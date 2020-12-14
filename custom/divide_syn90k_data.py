import os
import shutil

from sklearn.model_selection import train_test_split

parent_dir = os.path.join("data", "mjsynth")

os.mkdir(os.path.join(parent_dir, "..", "syn90k"))
os.mkdir(os.path.join(parent_dir, "..", "syn90k", "train"))
os.mkdir(os.path.join(parent_dir, "..", "syn90k", "test"))

for root, dirs, files in os.walk(parent_dir, topdown=True):
    if len(files) > 1:
        train, test = train_test_split(files, test_size=0.2, random_state=17)

        for train_image in train:
            shutil.copy(os.path.join(root, train_image),
                        os.path.join(parent_dir, "..", "syn90k", "train", train_image))

        for test_image in test:
            shutil.copy(os.path.join(root, test_image),
                        os.path.join(parent_dir, "..", "syn90k", "test", test_image))


