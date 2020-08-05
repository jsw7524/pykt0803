import matplotlib.pyplot as plt
import numpy as np

range1 = np.array([-1, 3])
p = 3
tmp=p*range1
plt.plot(range1, p*range1+5, c='green')
plt.show()