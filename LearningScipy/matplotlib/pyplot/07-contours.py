import matplotlib.pyplot as plt 
import numpy as np 

def f(x, y):
	# the height function
	return (1 - x/2 + x**5 + y**3) * np.exp(-x**2-y**2)

n = 256
x = np.linspace(-3, 3, n)
y = np.linspace(-3, 3, n)
print(x,y)
X, Y = np.meshgrid(x, y)
#plt.plot(X, Y, marker='.')
#print(np.meshgrid(x, y))
# 生成可以画网格的点
#          x1   x2   x3   x4
# [array([[-3., -1.,  1.,  3.],
#        [-3., -1.,  1.,  3.],
#        [-3., -1.,  1.,  3.],
#        [-3., -1.,  1.,  3.]]), 
# 		   y1   y2   y3  y4
# array([[-3., -3., -3., -3.],
#        [-1., -1., -1., -1.],
#        [ 1.,  1.,  1.,  1.],
#        [ 3.,  3.,  3.,  3.]])]
#plt.axes([0.025, 0.025, 0.95, 0.95])
C = plt.contour(X, Y, f(X,Y), 8, alpha=.75, cmap=plt.cm.hot)
# adding label
plt.clabel(C, inline=True, fontsize=10)

# plt.contourf(X, Y, f(X, Y), 8, alpha=.75, cmap='jet')
# C = plt.contour(X, Y, f(X, Y), 8, colors='black', linewidth=.5)
plt.show()
