import numpy as np
#
# #a = np.zeros((10, 2))
# a = np.ones((10,2))
# #a = np.array(range(0,20),(10,2))
# print(a)
# b = a.T
# print(b)
# c = b.view()
# print(c)
# d = np.reshape(b, (5, 4))
# print(d)
# e = np.reshape(b, (20,))
# print(e)
# f = np.reshape(b, (20, 1))
# print(f)
# g = np.reshape(b, (20, -1))
# print(g)
# h = np.reshape(b, (1, 20))
# print(h)
# i = np.reshape(b, (-1, 20))
# print(i)
# j = np.reshape(b, (-1, 10))
# print(j)
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# import numpy as np
#
# a = np.array([[1, 2], [3, 4]])
# b = a.view()
# c = a
# print(f"a={a}\n,b={b}\n,c={c}")
# b.shape = (4, -1)
# print(f"a={a}\n,b={b}\n,c={c}")
# c.shape = (-1, 4)
# print(f"a={a}\n,b={b}\n,c={c}")
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
import numpy as np


def printAll():
    print("a:\n", a)
    print("b:\n", b)
    print("c:\n", c)
    print("d:\n", d)


a = np.array([[1, 2], [3, 4]])
b = a
c = a.view()
d = a.copy()
printAll()
a.shape=(4,)
print("after a.shape=(4,) ")
printAll()
a[0] = 100
print("after a[0] assign to 100")
printAll()
