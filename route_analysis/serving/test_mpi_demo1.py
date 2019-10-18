import multiprocessing
import time


def func(msg):
    return multiprocessing.current_process().name + '-' + msg


if __name__ == "__main__":
    pool = multiprocessing.Pool(processes=4)  # 创建4个进程
    results = []
    for i in range(10):
        msg = "hello %d" % (i)
        results.append(pool.apply_async(func, (msg,)))
    pool.close()  # 关闭进程池，表示不能再往进程池中添加进程，需要在join之前调用
    pool.join()  # 等待进程池中的所有进程执行完毕
    print ("Sub-process(es) done.")

    for res in results:
        print (res.get())