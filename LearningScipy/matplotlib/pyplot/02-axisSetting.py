import matplotlib.pyplot as plt 
import numpy as np 

x = np.linspace(-3, 3, 50)
y1 = 2*x + 1
y2 = x**2

plt.figure()
plt.plot(x, y2, label='x')
plt.plot(x, y1, c='red', lw=1.0, ls='--', label='x^2')
plt.legend(loc='best')
# set x limits
plt.xlim((-1, 2))
plt.ylim((-2, 3))
plt.xlabel('X-axis')
plt.ylabel('Y-axis')

# set new ticks(记号)
new_ticks = np.linspace(-1, 2, 5)
# plt.xticks(new_ticks, [r'n', r'i', r'h', r'a', r'o'])
# set tick label
# plt.yticks([-2, -1.8, -1, 1.22, 3],
# 		   [r'$really\ bad$', r'$bad$', r'$normal$', r'$good$', r'$really\ good$'])
# to use '$ $' for math text and nice looking, e.g. '$\pi$'

# gca = 'get current axis'
ax = plt.gca()
ax.spines['right'].set_color('none') # it can change the axis's style
ax.spines['top'].set_color('none')

ax.xaxis.set_ticks_position('bottom') # 设置ticks摆放的位置
# ACCEPTS: [ 'top' | 'bottom' | 'both' | 'default' | 'none' ]

# 设置坐标轴在figure的显示效果(x轴, y轴以什么数值为基准)
ax.spines['bottom'].set_position(('data', 0))
ax.spines['left'].set_position(('data', 0))

# adding a legend(图例) 看9-10行
"""legend( handles=(line1, line2, line3),
           labels=('label1', 'label2', 'label3'),
           'upper right')
    The *loc* location codes are::
          'best' : 0,          (currently not supported for figure legends)
          'upper right'  : 1,
          'upper left'   : 2,
          'lower left'   : 3,
          'lower right'  : 4,
          'right'        : 5,
          'center left'  : 6,
          'center right' : 7,
          'lower center' : 8,
          'upper center' : 9,
          'center'       : 10,"""

plt.show()