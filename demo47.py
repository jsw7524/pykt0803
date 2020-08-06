import tensorflow as tf
import numpy as np

print(f"current tensorflow version={tf.__version__}")
a = np.array([5, 3, 8])
b = np.array([3, -1, 2])
c = np.add(a, b)
print(c)

a2 = tf.constant([5, 3, 8])
b2 = tf.constant([3, -1, 2])
c2 = tf.add(a2, b2)
print(c2)
print(c2.numpy())
c3 = tf.add(a, b)
print(f"c3type={type(c3)} value={c3}")