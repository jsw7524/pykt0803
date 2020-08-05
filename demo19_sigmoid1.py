import matplotlib.pyplot as plt
import numpy as np

x = np.arange(-10, 10, 0.1)
f = 1/(1+np.exp(-x))
plt.xlabel('observation')
plt.ylabel('P(y=1|observation)')
plt.plot(x,f)

plt.show()