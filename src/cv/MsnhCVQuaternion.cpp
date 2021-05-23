#include "Msnhnet/cv/MsnhCVQuaternion.h"
namespace Msnhnet
{
using namespace std;
QuaternionD::QuaternionD(const QuaternionD &q)
{
    this->_q0 = q._q0;
    this->_q1 = q._q1;
    this->_q2 = q._q2;
    this->_q3 = q._q3;
}

QuaternionD::QuaternionD(const double &q0, const double &q1, const double &q2, const double &q3)
    :_q0(q0),
      _q1(q1),
      _q2(q2),
      _q3(q3)
{

}

QuaternionD::QuaternionD(const std::vector<double> &val)
{
    setVal(val);
}

void QuaternionD::setVal(const std::vector<double> &val)
{
    if(val.size()!=4)
    {
        throw Exception(1,"[QuaternionD]: val size must = 4! \n", __FILE__, __LINE__, __FUNCTION__);
    }

    this->_q0 = val[0];
    this->_q1 = val[1];
    this->_q2 = val[2];
    this->_q3 = val[3];
}

std::vector<double> QuaternionD::getVal() const
{
    return {this->_q0,this->_q1,this->_q2,this->_q3};
}

double QuaternionD::mod() const
{
    return sqrt(this->_q0*this->_q0 + this->_q1*this->_q1 + this->_q2*this->_q2 + this->_q2*this->_q2);
}

QuaternionD QuaternionD::invert() const
{
    double mod = this->mod();
    return QuaternionD(this->_q0/mod, this->_q1/mod, this->_q2/mod, this->_q3/mod);
}

double QuaternionD::getQ0() const
{
    return _q0;
}

double QuaternionD::getQ1() const
{
    return _q1;
}

double QuaternionD::getQ2() const
{
    return _q2;
}

double QuaternionD::getQ3() const
{
    return _q3;
}

void QuaternionD::print()
{
    std::cout<<"[ "<<_q0<<", "<<_q1<<", "<<_q2<<", "<<_q3<<"] "<<std::endl;
}

string QuaternionD::toString()
{
    std::stringstream buf;
    buf<<"[ "<<_q0<<", "<<_q1<<", "<<_q2<<", "<<_q3<<"] "<<std::endl;
    return buf.str();
}

string QuaternionD::toHtmlString()
{
    std::stringstream buf;
    buf<<"[ "<<_q0<<", "<<_q1<<", "<<_q2<<", "<<_q3<<"] "<<"<br/>";
    return buf.str();
}

double QuaternionD::operator[](const uint8_t &index)
{
    if(index >4)
    {
        throw Exception(1,"[QuaternionD]: index out of memory! \n", __FILE__, __LINE__, __FUNCTION__);
    }
    if(index == 0)
    {
        return this->_q0;
    }
    else if(index == 1)
    {
        return this->_q1;
    }
    else if(index == 2)
    {
        return this->_q2;
    }
    else
    {
        return this->_q3;
    }
}

QuaternionD &QuaternionD::operator=(const QuaternionD &q)
{
    if(this!=&q)
    {
        this->_q0 = q._q0;
        this->_q1 = q._q1;
        this->_q2 = q._q2;
        this->_q3 = q._q3;
    }

    return *this;
}

bool QuaternionD::operator==(const QuaternionD &q)
{
    if(abs(this->_q0-q.getQ0())<MSNH_F64_EPS&&
            abs(this->_q1-q.getQ1())<MSNH_F64_EPS&&
            abs(this->_q2-q.getQ2())<MSNH_F64_EPS&&
            abs(this->_q3-q.getQ3())<MSNH_F64_EPS)
    {
        return true;
    }
    else
    {
        return false;
    }
}

QuaternionD operator/(const QuaternionD &A, const QuaternionD &B)
{
    return A*B.invert();
}

QuaternionD operator*(const QuaternionD &A, const QuaternionD &B)
{
    return QuaternionD(
                A.getQ0()*B.getQ0()-A.getQ1()*B.getQ1()-A.getQ2()*B.getQ2()-A.getQ3()*B.getQ3(),
                A.getQ0()*B.getQ1()+A.getQ1()*B.getQ0()+A.getQ2()*B.getQ3()-A.getQ3()*B.getQ2(),
                A.getQ0()*B.getQ2()-A.getQ1()*B.getQ3()+A.getQ2()*B.getQ0()+A.getQ3()*B.getQ1(),
                A.getQ0()*B.getQ3()+A.getQ1()*B.getQ2()-A.getQ2()*B.getQ1()+A.getQ3()*B.getQ0()
                );
}

QuaternionD operator+(const QuaternionD &A, const QuaternionD &B)
{
    return QuaternionD(A.getQ0()+B.getQ0(),
                       A.getQ1()+B.getQ1(),
                       A.getQ2()+B.getQ2(),
                       A.getQ3()+B.getQ3());
}

QuaternionD operator-(const QuaternionD &A, const QuaternionD &B)
{
    return QuaternionD(A.getQ0()-B.getQ0(),
                       A.getQ1()-B.getQ1(),
                       A.getQ2()-B.getQ2(),
                       A.getQ3()-B.getQ3());
}

QuaternionF::QuaternionF(const QuaternionF &q)
{
    this->_q0 = q._q0;
    this->_q1 = q._q1;
    this->_q2 = q._q2;
    this->_q3 = q._q3;
}

QuaternionF::QuaternionF(const float &q0, const float &q1, const float &q2, const float &q3)
    :_q0(q0),
      _q1(q1),
      _q2(q2),
      _q3(q3)
{

}

QuaternionF::QuaternionF(const std::vector<float> &val)
{
    setVal(val);
}

void QuaternionF::setVal(const std::vector<float> &val)
{
    if(val.size()!=4)
    {
        throw Exception(1,"[QuaternionF]: val size must = 4! \n", __FILE__, __LINE__, __FUNCTION__);
    }

    this->_q0 = val[0];
    this->_q1 = val[1];
    this->_q2 = val[2];
    this->_q3 = val[3];
}

std::vector<float> QuaternionF::getVal() const
{
    return {this->_q0,this->_q1,this->_q2,this->_q3};
}

float QuaternionF::mod() const
{
    return sqrt(this->_q0*this->_q0 + this->_q1*this->_q1 + this->_q2*this->_q2 + this->_q2*this->_q2);
}

QuaternionF QuaternionF::invert() const
{
    float mod = (float)this->mod();
    return QuaternionF(this->_q0/mod, this->_q1/mod, this->_q2/mod, this->_q3/mod);
}

void QuaternionF::print()
{
    std::cout<<"[ "<<_q0<<", "<<_q1<<", "<<_q2<<", "<<_q3<<"] "<<std::endl;
}

string QuaternionF::toString()
{
    std::stringstream buf;
    buf<<"[ "<<_q0<<", "<<_q1<<", "<<_q2<<", "<<_q3<<"] "<<std::endl;
    return buf.str();
}

string QuaternionF::toHtmlString()
{
    std::stringstream buf;
    buf<<"[ "<<_q0<<", "<<_q1<<", "<<_q2<<", "<<_q3<<"] "<<"<br/>";
    return buf.str();
}

float QuaternionF::operator[](const uint8_t &index)
{
    if(index >4)
    {
        throw Exception(1,"[QuaternionF]: index out of memory! \n", __FILE__, __LINE__, __FUNCTION__);
    }
    if(index == 0)
    {
        return this->_q0;
    }
    else if(index == 1)
    {
        return this->_q1;
    }
    else if(index == 2)
    {
        return this->_q2;
    }
    else
    {
        return this->_q3;
    }
}

QuaternionF &QuaternionF::operator=(const QuaternionF &q)
{
    if(this!=&q)
    {
        this->_q0 = q._q0;
        this->_q1 = q._q1;
        this->_q2 = q._q2;
        this->_q3 = q._q3;
    }

    return *this;
}

bool QuaternionF::operator ==(const QuaternionF &q)
{
    if(abs(this->_q0-q.getQ0())<MSNH_F32_EPS&&
            abs(this->_q1-q.getQ1())<MSNH_F32_EPS&&
            abs(this->_q2-q.getQ2())<MSNH_F32_EPS&&
            abs(this->_q3-q.getQ3())<MSNH_F32_EPS)
    {
        return true;
    }
    else
    {
        return false;
    }
}

QuaternionF operator-(const QuaternionF &A, const QuaternionF &B)
{
    return QuaternionF(A.getQ0()-B.getQ0(),
                       A.getQ1()-B.getQ1(),
                       A.getQ2()-B.getQ2(),
                       A.getQ3()-B.getQ3());
}

QuaternionF operator+(const QuaternionF &A, const QuaternionF &B)
{
    return QuaternionF(A.getQ0()+B.getQ0(),
                       A.getQ1()+B.getQ1(),
                       A.getQ2()+B.getQ2(),
                       A.getQ3()+B.getQ3());
}

QuaternionF operator*(const QuaternionF &A, const QuaternionF &B)
{
    return QuaternionF(
                A.getQ0()*B.getQ0()-A.getQ1()*B.getQ1()-A.getQ2()*B.getQ2()-A.getQ3()*B.getQ3(),
                A.getQ0()*B.getQ1()+A.getQ1()*B.getQ0()+A.getQ2()*B.getQ3()-A.getQ3()*B.getQ2(),
                A.getQ0()*B.getQ2()-A.getQ1()*B.getQ3()+A.getQ2()*B.getQ0()+A.getQ3()*B.getQ1(),
                A.getQ0()*B.getQ3()+A.getQ1()*B.getQ2()-A.getQ2()*B.getQ1()+A.getQ3()*B.getQ0()
                );
}

QuaternionF operator/(const QuaternionF &A, const QuaternionF &B)
{
    return A*B.invert();
}

float QuaternionF::getQ0() const
{
    return _q0;
}

float QuaternionF::getQ1() const
{
    return _q1;
}

float QuaternionF::getQ2() const
{
    return _q2;
}

float QuaternionF::getQ3() const
{
    return _q3;
}

}
