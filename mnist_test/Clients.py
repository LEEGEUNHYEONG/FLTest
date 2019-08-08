# %%
import matplotlib.pyplot as plt
import numpy as np

from tensorflow import keras

np.random.seed(42)

mnist = keras.datasets.mnist
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()


# %%    해당 이미지의 labels 별로 구분하여 리스트에 저장
'''
print("train image :", train_images.shape)
print("train label :", train_labels.shape)
print("text image : ", test_images.shape)
print("text label : ", test_labels.shape)
print('\n')
'''

index_list = [[], [], [], [], [], [], [], [], [], []]

for i, v in enumerate(train_labels):
    index_list[v].append(i)

plt.figure(figsize=(5, 5))
image = train_images[4]
plt.imshow(image, cmap='Greys')
plt.show()
