import matplotlib.pyplot as plt
import numpy as np

w1 = 3
b1 = -8.0
b2 = 0
b3 = 8
l1 = 'b=-8.0'
l2 = 'b=0'
l3 = 'b=8'
x = np.arange(-10, 10, 0.1)
for b, l in [(b1, l1), (b2, l2), (b3, l3)]:
    f = 1 / (1 + np.exp(-(x * w1 + b)))
    plt.plot(x, f, label=l)
plt.legend(loc=2)
plt.show()