#include <Msnhnet/robot/MsnhSpatialMath.h>

namespace Msnhnet
{

SO3D::SO3D(const Mat &mat)
{
    if(mat.getWidth()!=3 || mat.getHeight()!=3 || mat.getChannel()!=1 || mat.getStep()!=8 || mat.getMatType()!= MatType::MAT_GRAY_F64)

    {
        throw Exception(1, "[SO3D] mat should be: wxh==3x3 channel==1 step==8 matType==MAT_GRAY_F64", __FILE__, __LINE__,__FUNCTION__);
    }

    if(forceCheckSO3)
    {
        if(!mat.isRotMat())
        {
            throw Exception(1, "[SO3D] not a SO3 mat", __FILE__, __LINE__,__FUNCTION__);
        }
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

SO3D::SO3D(const SO3D &mat)
{
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

SO3D &SO3D::operator=(Mat &mat)
{
    if(mat.getWidth()!=3 || mat.getHeight()!=3 || mat.getChannel()!=1 || mat.getStep()!=8 || mat.getMatType()!= MatType::MAT_GRAY_F64)

    {
        throw Exception(1, "[SO3D] mat should be: wxh==3x3 channel==1 step==8 matType==MAT_GRAY_F64", __FILE__, __LINE__,__FUNCTION__);
    }

    if(forceCheckSO3)
    {
        if(!mat.isRotMat())
        {
            throw Exception(1, "[SO3D] not a SO3 mat", __FILE__, __LINE__,__FUNCTION__);
        }
    }

    if(this!=&mat)
    {
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
    return *this;
}

SO3D &SO3D::operator=(SO3D &mat)
{
    if(this!=&mat)
    {
        release();
        this->_channel  = mat._channel;
        this->_width    = mat._width;
        this->_height   = mat._height;
        this->_step     = mat._step;
        this->_matType  = mat._matType;

        if(mat._data.u8!=nullptr)
        {
            uint8_t *u8Ptr =  new uint8_t[this->_width*this->_height*this->_step]();
            memcpy(u8Ptr, mat._data.u8, this->_width*this->_height*this->_step);
            this->_data.u8 =u8Ptr;
        }
    }
    return *this;
}

RotationMatD &SO3D::toRotMat()
{
    return *this;
}

QuaternionD SO3D::toQuaternion()
{
    return Geometry::rotMat2Quaternion(*this);
}

EulerD SO3D::toEuler(const RotSequence &rotSeq)
{
    return Geometry::rotMat2Euler(*this,rotSeq);
}

RotationVecD SO3D::toRotVector()
{
    return Geometry::rotMat2RotVec(*this);
}

bool SO3D::isSO3(Mat mat)
{
    return mat.isRotMat();
}

SO3D SO3D::adjoint()
{
    return *this;
}

SO3D SO3D::rotX(float angleInRad)
{
    return  Geometry::rotX(angleInRad);
}

SO3D SO3D::rotY(float angleInRad)
{
    return  Geometry::rotY(angleInRad);
}

SO3D SO3D::rotZ(float angleInRad)
{
    return  Geometry::rotZ(angleInRad);
}

SO3D SO3D::fromRotMat(const RotationMatD &rotMat)
{
    return rotMat;
}

SO3D SO3D::fromQuaternion(const QuaternionD &quat)
{
    return Geometry::quaternion2RotMat(quat);
}

SO3D SO3D::fromEuler(const EulerD &euler, const RotSequence &rotSeq)
{
    return Geometry::euler2RotMat(euler,rotSeq);
}

SO3D SO3D::fromRotVec(const RotationVecD &rotVec)
{
    return Geometry::rotVec2RotMat(rotVec);
}

Matrix3x3D SO3D::wedge(const Vector3D &vec3)
{
    Matrix3x3D mat3x3;
    mat3x3.setVal({
                           0,  -vec3[2],    vec3[1],
                     vec3[2],      0   ,    -vec3[0],
                     -vec3[1], vec3[0],    0
                  });
    return mat3x3;
}

Vector3D SO3D::vee(const Matrix3x3D &mat3x3)
{
    Vector3D vec3d;
    vec3d[0] = 0.5*(mat3x3.getValAtRowCol(2,1) - mat3x3.getValAtRowCol(1,2));
    vec3d[1] = 0.5*(mat3x3.getValAtRowCol(0,2) - mat3x3.getValAtRowCol(2,0));
    vec3d[2] = 0.5*(mat3x3.getValAtRowCol(1,0) - mat3x3.getValAtRowCol(0,1));
    return vec3d;
}

SO3D SO3D::exp(const Vector3D &vec3)
{
    double angle = vec3.length();

    if(abs(angle)<MSNH_F64_EPS)

    {
        return Mat::eye(3,MAT_GRAY_F64) + wedge(vec3);
    }

    Vector3D axis = vec3/angle;

    double s = sin(angle);
    double c = cos(angle);

    Matrix3x3D skew = wedge(axis);

    return Mat::eye(3,MAT_GRAY_F64) + s*skew + (1-c)*skew*skew;

}

SO3D SO3D::exp(const Vector3D &vec3, double theta)
{
    return exp(vec3, theta);
}

Vector3D SO3D::log()
{
    if(*this == Mat::eye(3,MAT_GRAY_F64))
    {
        return Vector3D({0,0,0});
    }
    else if(abs(this->trace()+1)<MSNH_F64_EPS)
    {
        double m00 = this->getValAtRowCol(0,0);
        double m01 = this->getValAtRowCol(0,1);
        double m02 = this->getValAtRowCol(0,2);

        double m10 = this->getValAtRowCol(1,0);
        double m11 = this->getValAtRowCol(1,1);
        double m12 = this->getValAtRowCol(1,2);

        double m20 = this->getValAtRowCol(2,0);
        double m21 = this->getValAtRowCol(2,1);
        double m22 = this->getValAtRowCol(2,2);

        if(m00>m11 && m00>m22)
        {
            return 1/(std::sqrt(1+m00))*Vector3D({m00+1, m10, m20});
        }
        else if(m11>m00 && m11>m22)
        {
            return 1/(std::sqrt(1+m11))*Vector3D({m01+1, m11, m21});
        }
        else if(m22>m00 && m22>m11)
        {
            return 1/(std::sqrt(1+m22))*Vector3D({m02+1, m12, m22});
        }
    }
    else
    {
        double theta = std::acos(0.5*(this->trace()-1));
        return SO3D::vee(1/(2*std::sin(theta))*(*this - (*this).transpose()));
    }
}

}
