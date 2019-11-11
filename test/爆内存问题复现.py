import tensorflow as tf
import random
@tf.function
def add(a, b):
    return a + b + random.randint(0,1)


while(1):
    x = add(1, 2)
    print(x)
