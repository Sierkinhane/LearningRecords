'''
线程 -- 共享全局变量
        -- 不共享非全局变量
'''

import  threading
import time

def toil_and_moil():
    for i in range(5):
        print('thread-1')
        time.sleep(1)

def toil_and_moil2():
    for i in range(5):
        print('thread-2')
        time.sleep(1)

t1 = threading.Thread(target=toil_and_moil)
t2 = threading.Thread(target=toil_and_moil2)
t1.start()
t2.start()

# get the current number of threading
num = len(threading.enumerate())
print('thread num',num)

# the main thread run over after all sub-thread finished
print('main --- run over')
