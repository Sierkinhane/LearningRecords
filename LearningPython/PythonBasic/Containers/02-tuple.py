# tuples and sequences
# immutalbe
tup  = 1, 2, 3, (1, 2)
#print(tup)
tup_1 = 1, [2, 3], 3
tup_1[1][1] = 0
#print(tup_1)

# with none or one item
tup_2 = ()
#print(tup_2)
tup_3 = 1,
#print(tup_3)

# set--unordered and no duplicate elements and not support indexing
# 中文字典去重 1, 转化类型为set(去重) 2, ''.join() 拼接
str_1 ='查尔斯斯再次得的得分,与第一次破门如出一辙'
set_2 = set(str_1)
list_1 = list(set_2)
sequences = ''.join(list_1)
# exclude 得分 elements
set_2 = {x for x in set_2 if x not in '得分'}
#print(set_2)


# dictionary  keys are unique (within one dictionary)
# >>> tel = {'jack': 4098, 'sape': 4139}
# >>> tel['guido'] = 4127
# >>> tel
# {'jack': 4098, 'sape': 4139, 'guido': 4127}
# >>> tel['jack']
# 4098
# >>> del tel['sape']
# >>> tel['irv'] = 4127
# >>> tel
# {'jack': 4098, 'guido': 4127, 'irv': 4127}
# >>> list(tel)
# ['jack', 'guido', 'irv']
# >>> sorted(tel)
# ['guido', 'irv', 'jack']
# >>> 'guido' in tel
# True
# >>> 'jack' not in tel
# False

a = dict([('sape', 4139), ('guido', 4127), ('jack', 4098)])
#print(a)
dic_1 = {x : x**2 for x in [1, 2, 3]}
#print(dic_1)

dic_2 = dict(alen=2, peter=3, linda=4)
#print(dic_2)

# looping techniques
knights = {'gallahad': 'the pure', 'robin': 'the brave'}
for k, v in knights.items():
	#print(k, v)
	pass
for i, v in enumerate(['tic', 'tac', 'toe']):
	#print(i, v)
	pass

questions = ['name', 'quest', 'favorite color']
answers = ['lancelot', 'the holy grail', 'blue']
for q, a in zip(questions, answers):
    #print('What is your {0}?  It is {1}.'.format(q, a))
    pass
set_3 = {2, 1, 3}
a = sorted(set_3) # 自动变成list类型
print(a)
for i in reversed(range(1, 10, 2)):
    print(i)

# Comparing Sequences and Other Types
# (1, 2, 3)              < (1, 2, 4)
# [1, 2, 3]              < [1, 2, 4]
# 'ABC' < 'C' < 'Pascal' < 'Python'
# (1, 2, 3, 4)           < (1, 2, 4)
# (1, 2)                 < (1, 2, -1)
# (1, 2, 3)             == (1.0, 2.0, 3.0)
# (1, 2, ('aa', 'ab'))   < (1, 2, ('abc', 'a'), 4)

# a = ['a', 'b', 'c'] 遍历还打印它对应的index
# for i, v in enumerate(a):
# 	print(i, v)