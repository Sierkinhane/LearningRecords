from multiprocessing import Process
import os

def run():
    print('id:{0}'.format(os.getpid()))

if __name__ == '__main__':

    p = Process(target=run, args=())
    p.start()
    # p.terminate()
    p.join()