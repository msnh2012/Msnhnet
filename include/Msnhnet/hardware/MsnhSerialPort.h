#ifndef SERIALPORT_H
#define SERIALPORT_H
#include <string>
#include "Msnhnet/3rdparty/serial/serial.h"

namespace Msnhnet
{
class MsnhNet_API SerialPort: public Serial
{
public:
    static std::vector<Msnhnet::PortInfo> searchPorts();
    SerialPort(const std::string &port = "", uint32_t baudrate = 9600,Timeout timeout = Timeout(),bytesize_t bytesize = eightbits,
               parity_t parity = parity_none,stopbits_t stopbits = stopbits_one, flowcontrol_t flowcontrol = flowcontrol_none);
};
}

#endif 

