#include "Msnhnet/robot/MsnhFrame.h"

namespace Msnhnet
{

Frame::Frame(const Mat &mat)
{
    if(mat.getWidth()!=4 || mat.getHeight()!=4 || mat.getChannel()!=1 || mat.getStep()!=8 || mat.getMatType()!= MatType::MAT_GRAY_F64)

    {
        throw Exception(1, "[Frame] mat should be: wxh==4x4 channel==1 step==8 matType==MAT_GRAY_F64", __FILE__, __LINE__,__FUNCTION__);
    }
    release();
    this->_channel  = mat.getChannel();
    this->_width    = mat.getWidth();
    this->_height   = mat.getHeight();
    this->_step     = mat.getStep();
    this->_matType  = mat.getMatType();

    if(mat.getBytes()!=nullptr)
    {
        uint8_t *u8Ptr =  new uint8_t[this->_width*this->_height*this->_step]();
        memcpy(u8Ptr, mat.getBytes(), this->_width*this->_height*this->_step);
        this->_data.u8 =u8Ptr;
    }
}

Frame::Frame(const RotationMatD &rotMat)
{
    setRotationMat(rotMat);
}

Frame::Frame(const TranslationD &trans)
{
    setTranslation(trans);
}

Frame::Frame(const RotationMatD &rotMat, const TranslationD &trans)
{
    setRotationMat(rotMat);
    setTranslation(trans);
}

Frame Frame::fastInvert()
{
    RotationMatD rotMat = getRotationMat();
    TranslationD trans  = getTranslation();
    trans = trans * -1;

    rotMat = rotMat.transpose();
    trans = rotMat.mulVec(trans);

    return  Frame(rotMat,trans);
}

Frame Frame::SDH(double a, double alpha, double d, double theta)
{

    Frame frame;

    double ct   =   0;
    double st   =   0;
    double ca   =   0;
    double sa   =   0;

    ct = cos(theta);
    st = sin(theta);
    sa = sin(alpha);
    ca = cos(alpha);

    RotationMatD rotMat({   ct,    -st*ca,   st*sa,
                            st,     ct*ca,  -ct*sa,
                            0,        sa,      ca  });

    TranslationD trans({a*ct,   a*st,   d});

    frame.setRotationMat(rotMat);
    frame.setTranslation(trans);

    return frame;
}

Frame Frame::MDH(double a, double alpha, double d, double theta)
{
    Frame frame;

    double ct   =   0;
    double st   =   0;
    double ca   =   0;
    double sa   =   0;

    ct = cos(theta);
    st = sin(theta);
    sa = sin(alpha);
    ca = cos(alpha);

    RotationMatD rotMat({   ct,       -st,     0,
                            st*ca,  ct*ca,   -sa,
                            st*sa,  ct*sa,    ca });

    TranslationD trans({a,  -sa*d,  ca*d});

    frame.setRotationMat(rotMat);
    frame.setTranslation(trans);

    return frame;
}

QuaternionD Frame::getQuaternion() const
{
    return Geometry::rotMat2Quaternion(getRotationMat());
}

Frame &Frame::operator=(const Mat &mat)
{
    if(mat.getWidth()!=4 || mat.getHeight()!=4 || mat.getChannel()!=1 || mat.getStep()!=8 || mat.getMatType()!= MatType::MAT_GRAY_F64)

    {
        throw Exception(1, "[Frame] mat should be: wxh==4x4 channel==1 step==8 matType==MAT_GRAY_F64", __FILE__, __LINE__,__FUNCTION__);
    }

    if(this!=&mat)
    {
        release();
        this->_channel  = mat.getChannel();
        this->_width    = mat.getWidth();
        this->_height   = mat.getHeight();
        this->_step     = mat.getStep();
        this->_matType  = mat.getMatType();

        if(mat.getData().u8!=nullptr)
        {
            uint8_t *u8Ptr =  new uint8_t[this->_width*this->_height*this->_step]();
            memcpy(u8Ptr, mat.getData().u8, this->_width*this->_height*this->_step);
            this->_data.u8 =u8Ptr;
        }
    }
    return *this;
}

}

