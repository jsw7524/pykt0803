import tensorflow as tf
from datetime import datetime


# make a directory logs
@tf.function
def computeArea(sides):
    a = sides[:, 0]
    b = sides[:, 1]
    c = sides[:, 2]
    s = (a + b + c) / 2
    s = tf.add(tf.add(a,b),c)
    s = s/2
    areaSquare = s * (s - a) * (s - b) * (s - c)
    return areaSquare ** 0.5


stamp = datetime.now().strftime("%Y%M%d-%H%M")
logdir = 'logs/%s' % stamp
writer = tf.summary.create_file_writer(logdir)
tf.summary.trace_on(graph=True, profiler=True)
# pure tf function
print(computeArea(tf.constant([[3.0, 4.0, 5.0], [6.0, 6.0, 6.0]])))

with writer.as_default():
    tf.summary.trace_export(name="my_func_trace", step=0, profiler_outdir=logdir)
# print(computeArea(tf.constant([[3.0, 4.0, 5.0], [6.0, 6.0, 6.0]])).numpy())