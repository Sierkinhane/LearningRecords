import matplotlib.pyplot as plt 
import numpy as np 

x = np.linspace(-3, 3, 50)
y1 = 2*x + 1
y2 = x**2

plt.figure()    # create a figure
plt.plot(x, y1)

plt.figure(num=2, figsize=(8, 5))
plt.plot(x, y2)
# plot the second curve in this figure with certain parameters
# lw -- linewidth
# ls -- linestyle
plt.plot(x, y1, color='red', lw=1.0, ls='--')
plt.show()