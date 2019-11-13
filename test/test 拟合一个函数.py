import matplotlib.pyplot as plt  # 可视化模块
import numpy as np
from tensorflow import keras
import tensorflow as tf


np.random.seed(1337)  # for reproducibility

# create some data
X = np.linspace(-1, 1, 200)
np.random.shuffle(X)    # randomize the data
Y = 0.5 * X + 2 + np.random.normal(0, 0.05, (200, ))
# plot data
# plt.scatter(X, Y)
# plt.show()

X_train, Y_train = X[:160], Y[:160]     # train 前 160 data points
X_test, Y_test = X[160:], Y[160:]       # test 后 40 data points

model = keras.Sequential([
    keras.layers.Dense(32, activation='relu',
                         input_shape=(1,)),
    keras.layers.Dense(32, activation='relu'
                       ),
    keras.layers.Dense(1, name="test")
])


model.compile(optimizer='adam',
              loss='mse',
              metrics=['mse'])

model.summary()

# training
print('Training -----------')
for step in range(500):
    cost = model.train_on_batch(X_train, Y_train)
    # tf.keras.backend.clear_session()
    if step % 100 == 0:
        print('train cost: ', cost)

tf.keras.backend.clear_session()
# test
print('\nTesting ------------')
cost = model.evaluate(X_test, Y_test, batch_size=32)
print('test cost:', cost)
W, b = model.layers[0].get_weights()
print('Weights=', W, '\nbiases=', b)

@tf.function
def predict():
    model.predict(X_test)
    
for i in range(10000):
    predict()
    tf.keras.backend.clear_session()


# plotting the prediction
Y_pred = model.predict(X_test)
plt.scatter(X_test, Y_test)
plt.plot(X_test, Y_pred)
plt.show()
