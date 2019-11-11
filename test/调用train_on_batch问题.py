import tensorflow as tf
from tensorflow import keras
import numpy as np
import random
# fashion_mnist = keras.datasets.fashion_mnist

# (train_images, train_labels), (test_images,
#                                test_labels) = fashion_mnist.load_data()
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(4, )),
    keras.layers.Dense(32, activation='relu'),
    keras.layers.Dense(2)
])

model.compile(optimizer='adam',
              loss='mse',
              metrics=['accuracy'])

# print(model.predict(test_images))
# model.fit(train_images, train_labels, epochs=10)
# print(model.predict(test_images))

# test y = x1^2 + 3*x2 + x3 + 2*x4


def y(np_array):
    result = []
    if(np_array.shape == (1)):
        result = np_array[1]*np_array[1] + \
            np_array[2]*3+np_array[3]+np_array[4]*2
    else:
        for iter in np_array:
            result = np_array[1]*np_array[1] + \
            np_array[2]*3+np_array[3]+np_array[4]*2
    return result


test_data = np.array([
    y([random(),
    [4, 3, 2, 1],
    [4, 4, 3, 1]
])
test_data_label=before_predict=model.predict(test_data)
print("before: ", before_predict)
model.train_on_batch(
    test_data,
    np.array([[0.1, 0.7],
              [0.7, 0.1],
              [0.7, 0.2]])
)
after_predict=model.predict(test_data)
print("after: ", after_predict)

print("end")
