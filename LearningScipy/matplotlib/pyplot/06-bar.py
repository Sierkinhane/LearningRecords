import matplotlib.pyplot as plt 
import numpy as np 

n = 12
X = np.arange(n)
Y1 = ((1-X / float(n)) * np.random.uniform(0.5, 1.0, n))
Y2 = ((1-X / float(n)) * np.random.uniform(0.5, 1.0, n))
plt.bar(X, +Y1, facecolor='#9999ff', edgecolor='black')
plt.bar(X, -Y2, facecolor='#B41818', edgecolor='white')

for x, y in zip(X, Y1):
	# ha: horizontal alignment
	# va: vertical alignment
	plt.text(x+0.04, y+0.04, '%.2f' % y, ha='center', va='bottom')

for x, y in zip(X, Y2):
	plt.text(x+0.04, -y-0.1, '%.2f' % y, ha='center', va='bottom')

plt.xlim(-.5, n)
plt.xticks(())
plt.ylim(-1.25, 1.25)
plt.yticks(())
# fill_between
# n = 256
# X = np.linspace(-np.pi, np.pi, n, endpoint=True)
# Y = np.sin(2 * X)

# plt.fill_between(X, Y + 1, color='#ff9999', alpha=1.00)
# plt.fill_between(X, Y - 1, color='#9999ff', alpha=1.00)

plt.show()