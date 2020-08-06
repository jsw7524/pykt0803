
import tensorflow as tf

tf.compat.v1.disable_eager_execution()
hello = tf.constant("Hello tensorflow using 1.X!")
print(type(hello))
print(hello)
session1 = tf.compat.v1.Session()
print(session1.run(hello))
session1.close()