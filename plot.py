import matplotlib.pyplot as plt
import numpy as np
# multiple line plot


a = np.array([i for i in range(11)], dtype='float')
a = a / 25
print(a)
x = a/(1-a)
al = np.log10(1/x)
plt.plot( a, x,  marker='', color='black', linewidth=2, label="a/1-a")
plt.plot( a, al,  marker='', color='black', linewidth=2, label="alpha")
plt.xlabel('classifier error')
plt.ylabel('beta')
plt.show()
