import torch
import random
import numpy as np
import matplotlib.pyplot as plt
from som import SOM

# Set color dataset
colors = np.empty((0,3), float)
colors = np.append(colors, np.array([[0, 0, 0]]), axis=0)
colors = np.append(colors, np.array([[1, 1, 1]]), axis=0)
for i in range(10):
    colors = np.append(colors, np.array([[0, 0, random.random()]]), axis=0)
    colors = np.append(colors, np.array([[0, random.random(), 0]]), axis=0)
    colors = np.append(colors, np.array([[random.random(), 0, 0]]), axis=0)
    colors = np.append(colors, np.array([[1, 1, random.random()]]), axis=0)
    colors = np.append(colors, np.array([[1, random.random(), 1]]), axis=0)
    colors = np.append(colors, np.array([[random.random(), 1, 1]]), axis=0)
    colors = np.append(colors, np.array([[0, random.random(), random.random()]]), axis=0)
    colors = np.append(colors, np.array([[random.random(), random.random(), 0]]), axis=0)
    colors = np.append(colors, np.array([[1, random.random(), random.random()]]), axis=0)
    colors = np.append(colors, np.array([[random.random(), random.random(), 1]]), axis=0)
    colors = np.append(colors, np.array([[random.random(), random.random(), random.random()]]), axis=0)
data = torch.Tensor(colors)

row = 40
col = 40
total_epoch = 1000

som = SOM(3, (row, col))
for iter_no in range(total_epoch):
    som.self_organizing(data, iter_no, total_epoch)

weight = som.weight.reshape(3, row, col).numpy()
weight = np.transpose(weight, (1, 2, 0,))

plt.title('Color SOM')
plt.imshow(weight)
plt.show()
