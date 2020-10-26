#ifndef MSNHCVKALMAN_H
#define MSNHCVKALMAN_H

#include <Msnhnet/cv/MsnhCVMat.h>
namespace Msnhnet
{
class MsnhNet_API Kalman
{
public:
    explicit Kalman(const Mat &x, const Mat &P, const Mat &Q, const Mat &H, const Mat &R);
    ~Kalman();

    Mat getX() const;

    void setX(const Mat &x);

    Mat getF() const;

    void setF(const Mat &F);

    Mat getP() const;

    void setP(const Mat &P);

    Mat getQ() const;

    void setQ(const Mat &Q);

    Mat getH() const;

    void setH(const Mat &H);

    Mat getR() const;

    void setR(const Mat &R);

    void predict();

    void update(const Mat &z);

private:

    Mat _x;

    Mat _F;

    Mat _P;

    Mat _Q;

    Mat _H;

    Mat _R;

};

}

#endif 

