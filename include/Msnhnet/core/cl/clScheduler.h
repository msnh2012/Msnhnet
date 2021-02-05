#ifndef MSNHCLSCHEDULER_H
#define MSNHCLSCHEDULER_H

#include "Msnhnet/config/MsnhnetOpenCL.h"

namespace Msnhnet
{

class clScheduler 
{
public:
    clScheduler();
    clScheduler(const clScheduler &) = delete;
    clScheduler &operator=(const clScheduler &) = delete; 

    static clScheduler& get();
    void init();

private:
    
    cl::Context         _context;
    cl::CommandQueue    _queue;

}
}

#endif