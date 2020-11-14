#ifndef MSNHCVFILTERS_H
#define MSNHCVFILTERS_H

#include <vector>
#include <list>
#include <algorithm>
#include <Msnhnet/cv/MsnhCVMat.h>

namespace Msnhnet
{
class MsnhNet_API Filter1D
{
public:

    double ampLimLastVal=0;
    double ampFilter(double currentVal,double err);

    static double midFilter(double* data,int len);

    static double aveFilter(double* data,int len);

    int listAveSize=0;
    std::list<double> listAve;
    double listAveFilter(double currentVal);

    static double midAveFilter(double* data,int len);

    double ampAveLastVal=0;
    double ampAveFilter(double* data, int len, double err);

    double lastLagVal=0;
    double firstOrderLagFilter(double val, double a);

    double avoidWLastVal=0;
    double avoidWCnt=0;
    double avoidWiggleFilter(double val,int N);

    double ampAWLastVal=0;
    double ampAWCnt=0;
    double ampAWFilter(double val,double err, int N);

};

class MsnhNet_API KalmanFilter
{
public:
     KalmanFilter(const Mat &x, const Mat &P, const Mat &Q, const Mat &H, const Mat &R);
     KalmanFilter(){}
     ~KalmanFilter();

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

class MsnhNet_API SimpleKF1D
{
public:
    SimpleKF1D();

    void setF(const float &dt=0.01);
    void setX(const float &x = 0);
    void setP(const float &x = 1, const float &v = 100);
    void setQ();
    void setH();
    void setR(const float &err = 1000);
    void initKF();

    float update(const float &val);

private:
    KalmanFilter _kf;
};

}

#endif
