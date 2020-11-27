#ifndef MSNHCVSINGLETON_H
#define MSNHCVSINGLETON_H
#include <mutex>
namespace MsnhNet
{
template <class T>
class Singleton
{
public:
    static T* Instance()
    {
        static std::mutex mutex;
        static std::unique_ptr<T> inst;
        if (!inst)
        {
            mutex.lock();
            if (!inst) {
                inst.reset(new T);
            }
            mutex.unlock();
        }
        return inst.data();
    }
};
}
#endif 

