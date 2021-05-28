#include "Msnhnet/hardware/MsnhSerialPort.h"

namespace Msnhnet
{
std::vector<Msnhnet::PortInfo> SerialPort::searchPorts()
{
    return Msnhnet::list_ports();
}

SerialPort::SerialPort(const std::string &port, uint32_t baudrate, Timeout timeout, bytesize_t bytesize, parity_t parity, stopbits_t stopbits, flowcontrol_t flowcontrol)
    :Serial(port, baudrate, timeout, bytesize, parity, stopbits, flowcontrol)
{

}

}

