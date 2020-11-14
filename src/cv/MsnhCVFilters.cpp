#include <Msnhnet/cv/MsnhCVFilters.h>
namespace Msnhnet
{

double Filter1D::ampFilter(double currentVal, double err)
{
    if((currentVal-ampLimLastVal)>err||(ampLimLastVal-currentVal)>err)
    {

    }
    else
    {
        ampLimLastVal = currentVal;
    }
    return ampLimLastVal;
}

double Filter1D::midFilter(double *data, int len)
{

    std::vector<double> temp;

    for(int i =0;i<len;i++)
    {
        temp.push_back(data[i]);
    }

    std::sort(temp.begin(),temp.end());

    return temp[(len)*0.5];
}

double Filter1D::aveFilter(double *data, int len)
{
    double sum = 0;

    for(int i =0;i<len;i++)
    {
        sum+=data[i];
    }

    return sum/len;
}

double Filter1D::listAveFilter(double currentVal)
{
    if(listAve.size()>listAveSize)
    {
        listAve.pop_front();
    }

    listAve.push_back(currentVal);

    double sum = 0;

    for(auto i:listAve)
    {
        sum+=i;
    }

    return sum/listAveSize;
}

double Filter1D::midAveFilter(double *data, int len)
{
    std::list<double> temp;

    for(int i =0;i<len;i++)
    {
        temp.push_back(data[i]);
    }

    temp.sort();

    temp.pop_front(); 

    temp.pop_back();

    double tempSum=0;

    for(auto i:temp)
    {
        tempSum+=i;
    }

    return tempSum/(len-2);

}

double Filter1D::ampAveFilter(double *data, int len, double err)
{
    double sum = 0.0;

    for(int i=0;i<len;i++)
    {
        if((data[i]-ampAveLastVal)>err||(ampAveLastVal-data[i])>err)
        {

        }
        else
        {
            ampAveLastVal = data[i];
        }

        sum+=ampAveLastVal;
    }

    return sum/len;
}

double Filter1D::firstOrderLagFilter(double val, double a)
{
    lastLagVal = val*a+lastLagVal*(1-a);
    return lastLagVal;
}

double Filter1D::avoidWiggleFilter(double val, int N)
{
    if(val!=avoidWLastVal)
    {
        avoidWCnt++;
        if(avoidWCnt>N)
        {
            avoidWCnt=0;
            avoidWLastVal = val;
        }
    }
    return avoidWLastVal;
}

double Filter1D::ampAWFilter(double val, double err, int N)
{
    double temp=0;

    if((val-ampAWLastVal)>err||(ampAWLastVal-val)>err)
    {

    }
    else
    {
        temp = val;
    }

    if(temp!=ampAWLastVal)
    {
        ampAWCnt++;
        if(ampAWCnt>N)
        {
            ampAWCnt=0;
            ampAWLastVal = temp;
        }
    }

    return ampAWLastVal;
}

KalmanFilter::KalmanFilter(const Mat &x, const Mat &P, const Mat &Q, const Mat &H, const Mat &R)
    :_x(x),
     _P(P),
     _Q(Q),
     _H(H),
     _R(R)
{
}

KalmanFilter::~KalmanFilter()
{

}

Mat KalmanFilter::getX() const
{
    return _x;
}

void KalmanFilter::setX(const Mat &x)
{
    _x = x;
}

Mat KalmanFilter::getF() const
{
    return _F;
}

void KalmanFilter::setF(const Mat &F)
{
    _F = F;
}

Mat KalmanFilter::getP() const
{
    return _P;
}

void KalmanFilter::setP(const Mat &P)
{
    _P = P;
}

Mat KalmanFilter::getQ() const
{
    return _Q;
}

void KalmanFilter::setQ(const Mat &Q)
{
    _Q = Q;
}

Mat KalmanFilter::getH() const
{
    return _H;
}

void KalmanFilter::setH(const Mat &H)
{
    _H = H;
}

Mat KalmanFilter::getR() const
{
    return _R;
}

void KalmanFilter::setR(const Mat &R)
{
    _R = R;
}

void KalmanFilter::predict()
{

    this->_x =this->_F*this->_x;

    this->_P = this->_F*this->_P*this->_F.transpose() + this->_Q;
}

void KalmanFilter::update(const Mat &z)
{

    Mat y = z - this->_H*this->_x;

    Mat S = this->_H*this->_P*this->_H.transpose() + this->_R;

    Mat K = this->_P*this->_H.transpose()*S.invert();

    this->_x = this->_x +  K*y;
    Mat I = Mat::eye(this->_x.getDataNum(),this->_x.getMatType());

    this->_P = (I-K*this->_H)*this->_P;

}

SimpleKF1D::SimpleKF1D()
{
    initKF();
}

void SimpleKF1D::setF(const float &dt)
{
    float F1[] = {1,dt,0,1};
    Mat F(2,2,MAT_GRAY_F32,F1);
    this->_kf.setF(F);
}

void SimpleKF1D::setX(const float &x)
{
    float x1[] = {x,0};
    Mat X(1,2,MAT_GRAY_F32,x1);
    this->_kf.setX(X);
}

void SimpleKF1D::setP(const float &x, const float &v)
{
    float P1[] = {x,0,
                  0,v};
    Mat P(2,2,MAT_GRAY_F32,P1);
    this->_kf.setP(P);
}

void SimpleKF1D::setQ()
{
    Mat Q = Mat::eye(2,MAT_GRAY_F32);
    this->_kf.setQ(Q);
}

void SimpleKF1D::setH()
{
    float H1[] = {1,0};
    Mat H(2,1,MAT_GRAY_F32,H1);
    this->_kf.setH(H);
}

void SimpleKF1D::setR(const float &err)
{
    Mat R = Mat::diag(1,MAT_GRAY_F32,err);
    this->_kf.setR(R);
}

void SimpleKF1D::initKF()
{
    setF();
    setX();
    setP();
    setQ();
    setH();
    setR();
}

float SimpleKF1D::update(const float &val)
{
    this->_kf.predict();
    Mat z = Mat::diag(1,MAT_GRAY_F32, val);
    this->_kf.update(z);
    return this->_kf.getX().getFloat32()[0];
}

}
