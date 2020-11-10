#ifndef MSNHCVTHREAD_H
#define MSNHCVTHREAD_H
#include <thread>

class Thread {

public:
    void start()
    {
        uthread = new std::thread(Thread::exec, this);
    }

    void join() {
        uthread->join();
        delete uthread;
        uthread = NULL;
    }

    Thread(){}

    virtual ~Thread()
    {
        join();
        if (uthread) {
            delete uthread;
        }
    }

protected:

    virtual void run() = 0;

private:
    std::thread* uthread = NULL;

    static void exec(Thread* cppThread)
    {
        cppThread->run();
    }
};
#endif 

