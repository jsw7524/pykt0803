import tensorflow as tf

def computeArea(sides):
    a = sides[:, 0]
    b = sides[:, 1]
    c = sides[:, 2]

    s = (a + b + c) / 2
    areaSquare = s * (s - a) * (s - b) * (s - c)
    return areaSquare ** 0.5

with tf.compat.v1.Session() as session1:
    area = computeArea(tf.compat.v1.constant([[3.0, 4.0, 5.0],
                                              [6.0, 6.0, 6.0]]))
    result = session1.run(area)
    area2 = computeArea(tf.compat.v1.constant([[6.0, 8.0, 10.0],
                                               [12.0, 12.0, 12.0]]))
    result2 = session1.run(area2)
    print(result)
    print(result2)