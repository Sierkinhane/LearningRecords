# list methods
# 1.clear() and del
list_1 = [1, 2, 3]
list_1.clear() # [] clear the list_1
#print(list_1)
del list_1 # delete the list_1
#print(list_1)

# 2. index()
list_1 = [1, 2, 3, 2, 4]
index = list_1.index(2, 2, 4) # 寻找元素为2的下表，默认从index[0]开始，2, 4 --从下标为2到下标为4中寻找
#print(index)

# 3. count(x)
cnt = list_1.count(2) # return the number of times x appears in the list
#print(cnt)

# 4. sort(cmp=None, key=None, reverse=False) 
#	 sorted(iterable, cmp=None, key=None, reverse=False) 
#	 sort是容器的函数，sorted是Python的内建函数相同的参数
list_1 = [3, 4, 2, 4, 6, 8, 1]
list_1.sort() # 无返回值
#print(list_1)
sorteds = sorted(list_1) # 有返回值
#print(sorteds)

list_1=[(6,'cangjingkong',20),(4,'wutenglan',30),(7,'boduoyejiyi',25)]
list_1.sort()
#print(list_1) # 默认按照tuple[0]排序
sorteds = sorted(list_1)
#print(sorteds)

# key值排序
list_1.sort(key = lambda age : age[2])
#print(list_1)
sorteds = sorted(list_1, key = lambda age : age[2])
#print(sorteds)

# cmp值排序(已丢弃)
# reverse
list_1.sort(reverse = True)
#print(list_1)
list_1.reverse()
#print(list_1)

# Using lists as stacks
# Using lists as queues
from collections import deque
queue = deque([1, 2, 3, 4])
queue.append(5)
queue.popleft()
queue.popleft()
#print(queue)

# list comprehensions
# x从range()取值，执行x**2，append到list
squares = list(map(lambda x:x**2, range(2,10)))
print(squares)
squares = [x**2 for x in range(2,10)]
print(squares)
# it's equivalent to
# squares = []
# for x in range(2, 10):
# 	squares.append(x**2)

list_1 = [(x, y) for x in [1, 2, 3] for y in [2, 3, 4] if x != y]
print(list_1)

# >>> vec = [-4, -2, 0, 2, 4]
# >>> # create a new list with the values doubled

# >>> [x*2 for x in vec]
# [-8, -4, 0, 4, 8]

# >>> # filter the list to exclude negative numbers
# >>> [x for x in vec if x >= 0]
# [0, 2, 4]

# >>> # apply a function to all the elements
# >>> [abs(x) for x in vec]
# [4, 2, 0, 2, 4]

# >>> # call a method on each element
# >>> freshfruit = ['  banana', '  loganberry ', 'passion fruit  ']
# >>> [weapon.strip() for weapon in freshfruit]
# ['banana', 'loganberry', 'passion fruit']

# >>> # create a list of 2-tuples like (number, square)
# >>> [(x, x**2) for x in range(6)]
# [(0, 0), (1, 1), (2, 4), (3, 9), (4, 16), (5, 25)]

# >>> # the tuple must be parenthesized, otherwise an error is raised
# >>> [x, x**2 for x in range(6)]
#   File "<stdin>", line 1, in <module>
#     [x, x**2 for x in range(6)]
#                ^
# SyntaxError: invalid syntax

# >>> # flatten a list using a listcomp with two 'for'
# >>> vec = [[1,2,3], [4,5,6], [7,8,9]]
# >>> [num for elem in vec for num in elem]
# [1, 2, 3, 4, 5, 6, 7, 8, 9]

# >>> from math import pi
# >>> [str(round(pi, i)) for i in range(1, 6)]
# ['3.1', '3.14', '3.142', '3.1416', '3.14159']
# pi = 1.22223
# print(round(pi, 2)) # round(number, i) 取小数点后几位，整数也不会报错

# 5.extend(iterable) no return
a = [1, 2, 3, 4]
b = [4, 5, 6, 7]
a.extend(b)
print(a)
