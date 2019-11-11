import tensorflow as tf
from tensorflow import keras


(x_train, y_train), (x_test, y_test) = keras.datasets.boston_housing.load_data()
print(x_train.shape, ' ', y_train.shape)
print(x_test.shape, ' ', y_test.shape)
# 构建模型

model = keras.Sequential([
    keras.layers.Dense(32, activation='relu', input_shape=(4,)),
    
    keras.layers.Dense(32, activation='relu'),
    keras.layers.Dense(2)
])

# 配置模型
model.compile(optimizer=keras.optimizers.SGD(0.1),
              loss='mean_squared_error',  # keras.losses.mean_squared_error
              metrics=['mse'])
model.summary()

# value = model.predict([1,2,3,4])
value = model.predict([[1,2,3,4]])


print("end")