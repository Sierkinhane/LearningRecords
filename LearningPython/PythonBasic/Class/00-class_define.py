'''
定义类有两种方法：
	1. 新式类Car(Object)
	2. 经典类Car
类命名方法：大驼峰
'''

class Car(object):

	# 创建完对象后会自动被调用
	def __init__(self, price, name):
		super(Car, self).__init__()
		self.price = price
		self.__name = name # 私有属性不能随意修改

	# 默认是返回对象在内存中的地址
	def __str__(self):
		info = '你好！'
		return info

	def getName(self):
		return self.__name

	# 当对象被删除后会自动被调用
	# 当有1个变量保存了对象的引用时，此对象的引用计数就会加1
	# 当使用del删除变量指向的对象时，如果对象的引用计数不会1，
	# 比如3，那么此时只会让这个引用计数减1，即变为2，当再次调
	# 用del时，变为1，如果再调用1次del，此时会真的把对象进行删除
	def __del__(self):
		print('bye bye!')

BMW = Car(400000, 'x6')
BMW2 = BMW
print(BMW.getName())
#print(BMW.__name) # AttributeError: 'Car' object has no attribute '__name'
del BMW # bye bye!
#del BMW2

'''
单继承
	1.子类没有定义__init__方法，但是父类有，所以在子类继承父类的时候
	这个方法就被继承了，所以只要创建Bosi的对象，就默认执行了那个继承过
	来的__init__方法

	2.子类在继承的时候，在定义类时，小括号()中为父类的名字
	父类的属性、方法，会被继承给子类

	3.私有的属性，不能通过对象直接访问，但是可以通过方法访问
	私有的方法，不能通过对象直接访问
	私有的属性、方法，不会被子类继承，也不能被访问
	一般情况下，私有的属性、方法都是不对外公布的，往往用来做内部的事情，起到安全的作用
'''
'''
如果在下面的多继承例子中，如果父类A和父类B中，有一个同名的方法，那么通过子类去调用的时候，调用哪个？
'''
class base(object):
    def test(self):
        print('----base test----')
class A(base):
    def test(self):
        print('----A test----')

# 定义一个父类
class B(base):
    def test(self):
        print('----B test----')

# 定义一个子类，继承自A、B
class C(A,B):
    pass


obj_C = C()
obj_C.test()

print(C.__mro__) #可以查看C类的对象搜索方法时的先后顺序
				 #C-->A-->B--base


'''
调用父类方法
'''
class Cat(object):
    def __init__(self,name):
        self.name = name
        self.color = 'yellow'


class Bosi(Cat):

    def __init__(self,name):
        # 调用父类的__init__方法1(python2)
        #Cat.__init__(self,name)
        # 调用父类的__init__方法2
        #super(Bosi,self).__init__(name)
        # 调用父类的__init__方法3
        super().__init__(name)

    def getName(self):
        return self.name

bosi = Bosi('xiaohua')

print(bosi.name)
print(bosi.color)

'''
如果需要在类外修改类属性，必须通过类对象去引用然后进行修改。
如果通过实例对象去引用，会产生一个同名的实例属性，这种方式修
改的是实例属性，不会影响到类属性，并且之后如果通过实例对象去
引用该名称的属性，实例属性会强制屏蔽掉类属性，即引用的是实例
属性，除非删除了该实例属性。
'''

'''
从类方法和实例方法以及静态方法的定义形式就可以看出来，类方法的
第一个参数是类对象cls，那么通过cls引用的必定是类对象的属性和方
法；而实例方法的第一个参数是实例对象self，那么通过self引用的可
能是类属性、也有可能是实例属性（这个需要具体分析），不过在存在相
同名称的类属性和实例属性的情况下，实例属性优先级更高。静态方法中
不需要额外定义参数，因此在静态方法中引用类属性的话，必须通过类
对象来引用
'''

'''
异常
'''