import matplotlib.pyplot as plt
import numpy as np

DATASET_TYPE = "Syn90K"

with open("data", "sar-results", "statistics.txt", "r") as file:
    data = file.read()
    data = data.split("\n")

accuracy = np.array([[float(value) for value in line.split(" ")] for line in data])
plt.plot(accuracy[:, 0], color="blue", label="Train")
plt.plot(accuracy[:, 1], color="orange", label="Test")
plt.title(DATASET_TYPE)
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()
plt.show()