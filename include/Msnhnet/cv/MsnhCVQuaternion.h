#ifndef MSNHCVQUATERNION_H
#define MSNHCVQUATERNION_H

#include "Msnhnet/config/MsnhnetCfg.h"
#include <iostream>
#include <sstream>
#include <iomanip>
#include <string>

namespace Msnhnet
{
class MsnhNet_API QuaternionD
{
public:
    QuaternionD(){}
    QuaternionD(const QuaternionD &q);
    QuaternionD(const double& q0, const double& q1, const double& q2, const double& q3);
    QuaternionD(const std::vector<double> &val);

    void setVal(const std::vector<double> &val);

    std::vector<double> getVal() const;

    double mod() const;

    QuaternionD invert() const;

    double getQ0() const;
    double getQ1() const;
    double getQ2() const;
    double getQ3() const;

    void print();

    std::string toString();

    std::string toHtmlString();

    double operator[] (const uint8_t& index);

    QuaternionD& operator=(const QuaternionD& q);

    bool operator== (const QuaternionD& q);

    MsnhNet_API friend QuaternionD operator- (const QuaternionD &A, const QuaternionD &B);
    MsnhNet_API friend QuaternionD operator+ (const QuaternionD &A, const QuaternionD &B);
    MsnhNet_API friend QuaternionD operator* (const QuaternionD &A, const QuaternionD &B);
    MsnhNet_API friend QuaternionD operator/ (const QuaternionD &A, const QuaternionD &B);
private:
    double _q0 = 0;
    double _q1 = 0;
    double _q2 = 0;
    double _q3 = 0;
};

class MsnhNet_API QuaternionF
{
public:
    QuaternionF(){}
    QuaternionF(const QuaternionF &q);
    QuaternionF(const float& q0, const float& q1, const float& q2, const float& q3);
    QuaternionF(const std::vector<float> &val);

    void setVal(const std::vector<float> &val);

    std::vector<float> getVal() const;

    float mod() const;

    QuaternionF invert() const;

    void print();

    std::string toString();

    std::string toHtmlString();

    float operator[] (const uint8_t& index);

    QuaternionF& operator=(const QuaternionF& q);

    bool operator ==(const QuaternionF& q);

    MsnhNet_API friend QuaternionF operator- (const QuaternionF &A, const QuaternionF &B);
    MsnhNet_API friend QuaternionF operator+ (const QuaternionF &A, const QuaternionF &B);
    MsnhNet_API friend QuaternionF operator* (const QuaternionF &A, const QuaternionF &B);
    MsnhNet_API friend QuaternionF operator/ (const QuaternionF &A, const QuaternionF &B);

    float getQ0() const;
    float getQ1() const;
    float getQ2() const;
    float getQ3() const;

private:
    float _q0 = 0;
    float _q1 = 0;
    float _q2 = 0;
    float _q3 = 0;
};

}

#endif 

