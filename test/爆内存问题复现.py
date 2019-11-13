import tensorflow as tf
import random
@tf.function
def add(a, b):
    return a + b + random.randint(0, 1)


def normaladd(a, b):
    return a+b


while(1):
    tf.keras.backend.clear_session()
    x = add(1, 2)
    # print(x)
    y = normaladd(1, 2)
