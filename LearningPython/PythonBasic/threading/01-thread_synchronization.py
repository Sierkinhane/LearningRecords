'''
互斥锁：
当一个线程调用锁的acquire()方法获得锁时，锁就进入“locked”状态。
每次只有一个线程可以获得锁。如果此时另一个线程试图获得这个锁，
该线程就会变为“blocked”状态，称为“阻塞”，直到拥有锁的线程调用
锁的release()方法释放锁之后，锁进入“unlocked”状态。线程调度程
序从处于同步阻塞状态的线程中选择一个来获得锁，并使得该线程进
入运行（running）状态。
'''

import threading

num = 0

def add_1():
    global num
    for i in range(1000000):
        mutex_flag = mutex.acquire(True)
        if mutex_flag:
            num += 1
            mutex.release()
    print('add1', num)

def add_2():
    global num
    for i in range(1000000):
        # True表示堵塞，即如果这个锁在上锁之前已经被上锁，那么这个线程会在这里一直等待
        # False表示非堵塞，无论是否成功上锁，都会往下执行
        mutex_flag = mutex.acquire(True)
        if mutex_flag:
            num += 1
            mutex.release()
    print('add2', num)

# 创建一个互斥锁,默认是未上锁的状态
mutex = threading.Lock()

t1 = threading.Thread(target=add_1, args=())
t2 = threading.Thread(target=add_2, args=())
t1.start()
t2.start()


print('main print num', num)
