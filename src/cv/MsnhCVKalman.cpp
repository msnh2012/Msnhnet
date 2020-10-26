#include <Msnhnet/cv/MsnhCVKalman.h>
namespace Msnhnet
{

Kalman::Kalman(const Mat &x, const Mat &P, const Mat &Q, const Mat &H, const Mat &R)
    :_x(x),
     _P(P),
     _Q(Q),
     _H(H),
     _R(R)
{
}

Kalman::~Kalman()
{

}

Mat Kalman::getX() const
{
    return _x;
}

void Kalman::setX(const Mat &x)
{
    _x = x;
}

Mat Kalman::getF() const
{
    return _F;
}

void Kalman::setF(const Mat &F)
{
    _F = F;
}

Mat Kalman::getP() const
{
    return _P;
}

void Kalman::setP(const Mat &P)
{
    _P = P;
}

Mat Kalman::getQ() const
{
    return _Q;
}

void Kalman::setQ(const Mat &Q)
{
    _Q = Q;
}

Mat Kalman::getH() const
{
    return _H;
}

void Kalman::setH(const Mat &H)
{
    _H = H;
}

Mat Kalman::getR() const
{
    return _R;
}

void Kalman::setR(const Mat &R)
{
    _R = R;
}

void Kalman::predict()
{

    this->_x = Mat::mul(this->_F,this->_x);

    this->_P = Mat::mul(Mat::mul(this->_F,this->_P),this->_F.transpose()) + this->_Q;
}

void Kalman::update(const Mat &z)
{

    Mat y = z - Mat::mul(this->_H,this->_x);

    Mat S = Mat::mul(Mat::mul(this->_H,this->_P),this->_H.transpose()) + this->_R;

    Mat K = Mat::mul(Mat::mul(this->_P,this->_H.transpose()),S.invert());

    this->_x = this->_x +  Mat::mul(K,y);
    Mat I = Mat::eye(this->_x.getDataNum(),this->_x.getMatType());

    this->_P = Mat::mul((I-Mat::mul(K,this->_H)),this->_P);

}
}
