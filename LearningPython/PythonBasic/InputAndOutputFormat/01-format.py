import math
print("网站名：{name}, 地址 {url}".format(name="菜鸟教程", url="www.runoob.com"))

print('{0},{1}'.format('zhangk', 32))

print('{},{},{}'.format('zhangk','boy',32))

print('{name},{sex},{age}'.format(age=32,sex='male',name='zhangk'))

# .3f精度，保留三位小数，四舍五入
print('{:.3f}'.format(31.31412))

# 填充与对齐
# 填充常跟对齐一起使用
# ^、<、>分别是居中、左对齐、右对齐，后面带宽度
# :号后面带填充的字符，只能是一个字符，不指定的话默认是用空格填充
 
print('{:>8}'.format('zhang'))
print('{:0>8}'.format('zhang'))
print('{:a<8}'.format('zhang'))
print('{:^10}'.format('zhang'))
print('{:^10}'.format('Inhane'))

# 其他类型
# 主要就是进制了，b、d、o、x分别是二进制、十进制、八进制、十六进制
print('{:b}'.format(15))
 
print('{:d}'.format(15))
 
print('{:o}'.format(15))
 
print('{:x}'.format(15))

# 用逗号还能用来做金额的千位分隔符
print('{:,}'.format(123456789))
 

table = {'Sjoerd': 4127, 'Jack': 4098, 'Dcab': 7678}
for name, phone in table.items():
    print('name {0:10} ====> phone {1:10}'.format(name, phone))

print('The value of pi is approximately {:.3f}'.format(math.pi))