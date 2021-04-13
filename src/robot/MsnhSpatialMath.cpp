#include <Msnhnet/robot/MsnhSpatialMath.h>

namespace Msnhnet
{

bool SO3D::forceCheckSO3 = true;
bool SO3F::forceCheckSO3 = true;
bool SE3D::forceCheckSE3 = true;

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

bool SO3D::isSO3(const Mat &mat)
{
    return mat.isRotMat();
}

SO3D SO3D::adjoint()
{
    return *this;
}

SO3D SO3D::rotX(double angleInRad)
{
    return  Geometry::rotX(angleInRad);
}

SO3D SO3D::rotY(double angleInRad)
{
    return  Geometry::rotY(angleInRad);
}

SO3D SO3D::rotZ(double angleInRad)
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

Matrix3x3D SO3D::wedge(const Vector3D &omg, bool needCalUnit)
{
    Vector3D tmp = omg;

    if(needCalUnit)
    {
        if(closeToZeroD(tmp.length()))
        {
            return Mat(3,3,MAT_GRAY_F64);
        }

        tmp = tmp / tmp.length();
    }

    Matrix3x3D mat3x3;
    mat3x3.setVal({
                            0,  -tmp[2],   tmp[1],
                       tmp[2],       0 ,  -tmp[0],
                      -tmp[1],   tmp[0],    0
                  });
    return mat3x3;
}

Vector3D SO3D::vee(const Matrix3x3D &mat3x3, bool needCalUnit)
{
    Vector3D omg;
    omg[0] = 0.5*(mat3x3.getValAtRowCol(2,1) - mat3x3.getValAtRowCol(1,2));
    omg[1] = 0.5*(mat3x3.getValAtRowCol(0,2) - mat3x3.getValAtRowCol(2,0));
    omg[2] = 0.5*(mat3x3.getValAtRowCol(1,0) - mat3x3.getValAtRowCol(0,1));

    if(needCalUnit)
    {
        if(closeToZeroD(omg.length()))
        {
            return Vector3D({0,0,0});
        }

        omg = omg / omg.length();
    }

    return omg;
}

SO3D SO3D::exp(const Vector3D &omg)
{
    double angle = omg.length();

    if(closeToZeroD(angle))

    {
        return Mat::eye(3,MAT_GRAY_F64) + wedge(omg);
    }

    Vector3D axis = omg/angle;

    double s = sin(angle);
    double c = cos(angle);

    Matrix3x3D skew = wedge(axis);

    return Mat::eye(3,MAT_GRAY_F64) + s*skew + (1-c)*skew*skew;

}

SO3D SO3D::exp(const Vector3D &omg, double theta)
{

    if(!closeToZeroD(omg.length()-1) && !closeToZeroD(omg.length()))
    {
        throw Exception(1, "[SO3D] given theta, OMG must be a unit vector", __FILE__, __LINE__,__FUNCTION__);
    }

    return SO3D::exp(omg*theta);
}

Vector3D SO3D::log()
{
    if(*this == Mat::eye(3,MAT_GRAY_F64))
    {
        return Vector3D({0,0,0});
    }
    else if(closeToZeroD(this->trace()+1))
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
            return 1/(sqrt(1+m00))*Vector3D({m00+1, m10, m20});
        }
        else if(m11>m00 && m11>m22)
        {
            return 1/(sqrt(1+m11))*Vector3D({m01+1, m11, m21});
        }
        else if(m22>m00 && m22>m11)
        {
            return 1/(sqrt(1+m22))*Vector3D({m02+1, m12, m22});
        }
    }
    else
    {
        double theta = acos(0.5*(this->trace()-1));
        return SO3D::vee(1.0/(2*sin(theta))*(*this - (*this).transpose()))*theta;
    }
}

SO3F::SO3F(const Mat &mat)
{
    if(mat.getWidth()!=3 || mat.getHeight()!=3 || mat.getChannel()!=1 || mat.getStep()!=4 || mat.getMatType()!= MatType::MAT_GRAY_F32)

    {
        throw Exception(1, "[SO3F] mat should be: wxh==3x3 channel==1 step==4 matType==MAT_GRAY_F32", __FILE__, __LINE__,__FUNCTION__);
    }

    if(forceCheckSO3)
    {
        if(!mat.isRotMat())
        {
            throw Exception(1, "[SO3F] not a SO3 mat", __FILE__, __LINE__,__FUNCTION__);
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

SO3F::SO3F(const SO3F &mat)
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

SO3F &SO3F::operator=(Mat &mat)
{
    if(mat.getWidth()!=3 || mat.getHeight()!=3 || mat.getChannel()!=1 || mat.getStep()!=4 || mat.getMatType()!= MatType::MAT_GRAY_F32)

    {
        throw Exception(1, "[SO3F] mat should be: wxh==3x3 channel==1 step==4 matType==MAT_GRAY_F32", __FILE__, __LINE__,__FUNCTION__);
    }

    if(forceCheckSO3)
    {
        if(!mat.isRotMat())
        {
            throw Exception(1, "[SO3F] not a SO3 mat", __FILE__, __LINE__,__FUNCTION__);
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

SO3F &SO3F::operator=(SO3F &mat)
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

RotationMatF &SO3F::toRotMat()
{
    return *this;
}

QuaternionF SO3F::toQuaternion()
{
    return Geometry::rotMat2Quaternion(*this);
}

EulerF SO3F::toEuler(const RotSequence &rotSeq)
{
    return Geometry::rotMat2Euler(*this,rotSeq);
}

RotationVecF SO3F::toRotVector()
{
    return Geometry::rotMat2RotVec(*this);
}

bool SO3F::isSO3(const Mat &mat)
{
    return mat.isRotMat();
}

SO3F SO3F::adjoint()
{
    return *this;
}

SO3F SO3F::rotX(float angleInRad)
{
    return  Geometry::rotX(angleInRad);
}

SO3F SO3F::rotY(float angleInRad)
{
    return  Geometry::rotY(angleInRad);
}

SO3F SO3F::rotZ(float angleInRad)
{
    return  Geometry::rotZ(angleInRad);
}

SO3F SO3F::fromRotMat(const RotationMatF &rotMat)
{
    return rotMat;
}

SO3F SO3F::fromQuaternion(const QuaternionF &quat)
{
    return Geometry::quaternion2RotMat(quat);
}

SO3F SO3F::fromEuler(const EulerF &euler, const RotSequence &rotSeq)
{
    return Geometry::euler2RotMat(euler,rotSeq);
}

SO3F SO3F::fromRotVec(const RotationVecF &rotVec)
{
    return Geometry::rotVec2RotMat(rotVec);
}

Matrix3x3F SO3F::wedge(const Vector3F &omg, bool needCalUnit)
{

    Vector3F tmp = omg;

    if(needCalUnit)
    {
        if(closeToZeroF((float)tmp.length()))
        {
            return Mat(3,3,MAT_GRAY_F32);
        }

        tmp = tmp / (float)tmp.length();
    }

    Matrix3x3F mat3x3;
    mat3x3.setVal({
                            0,   -tmp[2],    tmp[1],
                      tmp[2],          0,    -tmp[0],
                      -tmp[1],   tmp[0],    0
                  });
    return mat3x3;
}

Vector3F SO3F::vee(const Matrix3x3F &mat3x3, bool needCalUnit)
{
    Vector3F omg;
    omg[0] = 0.5f*(mat3x3.getValAtRowCol(2,1) - mat3x3.getValAtRowCol(1,2));
    omg[1] = 0.5f*(mat3x3.getValAtRowCol(0,2) - mat3x3.getValAtRowCol(2,0));
    omg[2] = 0.5f*(mat3x3.getValAtRowCol(1,0) - mat3x3.getValAtRowCol(0,1));

    if(needCalUnit)
    {
        if(closeToZeroF((float)omg.length()))
        {
            return Vector3F({0,0,0});
        }

        omg = omg / (float)omg.length();
    }
    return omg;
}

SO3F SO3F::exp(const Vector3F &omg)
{
    float angle = (float)omg.length();

    if(closeToZeroF(angle))

    {
        return Mat::eye(3,MAT_GRAY_F32) + wedge(omg);
    }

    Vector3F axis = omg/angle;

    float s = sinf(angle);
    float c = cosf(angle);

    Matrix3x3F skew = wedge(axis);

    return Mat::eye(3,MAT_GRAY_F32) + s*skew + (1-c)*skew*skew;

}

SO3F SO3F::exp(const Vector3F &omg, float theta)
{

    if(!closeToZeroF(omg.length()-1) && !closeToZeroF(omg.length()))
    {
        throw Exception(1, "[SO3F] given theta, OMG must be a unit vector", __FILE__, __LINE__,__FUNCTION__);
    }
    return SO3F::exp(omg*theta);
}

Vector3F SO3F::log()
{
    if(*this == Mat::eye(3,MAT_GRAY_F32))
    {
        return Vector3F({0,0,0});
    }
    else if(closeToZeroF(this->trace()+1))
    {
        float m00 = this->getValAtRowCol(0,0);
        float m01 = this->getValAtRowCol(0,1);
        float m02 = this->getValAtRowCol(0,2);

        float m10 = this->getValAtRowCol(1,0);
        float m11 = this->getValAtRowCol(1,1);
        float m12 = this->getValAtRowCol(1,2);

        float m20 = this->getValAtRowCol(2,0);
        float m21 = this->getValAtRowCol(2,1);
        float m22 = this->getValAtRowCol(2,2);

        if(m00>m11 && m00>m22)
        {
            return 1/(sqrtf(1+m00))*Vector3F({m00+1, m10, m20});
        }
        else if(m11>m00 && m11>m22)
        {
            return 1/(sqrtf(1+m11))*Vector3F({m01+1, m11, m21});
        }
        else if(m22>m00 && m22>m11)
        {
            return 1/(sqrtf(1+m22))*Vector3F({m02+1, m12, m22});
        }
    }
    else
    {
        float theta = acosf(0.5f*((float)this->trace()-1));
        return SO3F::vee(1/(2*sin(theta))*(*this - (*this).transpose()))*theta;
    }
}

SE3D::SE3D(const Mat &mat)
{
    if(mat.getWidth()!=4 || mat.getHeight()!=4 || mat.getChannel()!=1 || mat.getStep()!=8 || mat.getMatType()!= MatType::MAT_GRAY_F64)

    {
        throw Exception(1, "[SE3D] mat should be: wxh==4x4 channel==1 step==8 matType==MAT_GRAY_F64", __FILE__, __LINE__,__FUNCTION__);
    }

    if(forceCheckSE3)
    {
        if(!mat.isHomTransMatrix())
        {
            throw Exception(1, "[SE3D] not a SE3 mat", __FILE__, __LINE__,__FUNCTION__);
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

SE3D::SE3D(const SE3D &mat)
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

SE3D &SE3D::operator=(Mat &mat)
{
    if(mat.getWidth()!=4 || mat.getHeight()!=4 || mat.getChannel()!=1 || mat.getStep()!=8 || mat.getMatType()!= MatType::MAT_GRAY_F64)

    {
        throw Exception(1, "[SE3D] mat should be: wxh==4x4 channel==1 step==8 matType==MAT_GRAY_F64", __FILE__, __LINE__,__FUNCTION__);
    }

    if(forceCheckSE3)
    {
        if(!mat.isHomTransMatrix())
        {
            throw Exception(1, "[SE3D] not a SE3 mat", __FILE__, __LINE__,__FUNCTION__);
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

SE3D &SE3D::operator=(SE3D &mat)
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

Matrix4x4D &SE3D::toMatrix4x4()
{
    return *this;
}

Mat SE3D::adjoint()
{
    RotationMatD R = getRotationMat();
    TranslationD p = getTranslation();

    Mat rightUp = SO3D::wedge(p)*R;
    Mat zeros3x3= Mat(3,3,MAT_GRAY_F64);

    Mat up      = MatOp::vContact(R,rightUp);
    Mat down    = MatOp::vContact(zeros3x3,R);

    return MatOp::hContact(up,down);
}

Matrix4x4D SE3D::wedge(const ScrewD &screw, bool needCalUnit)
{
    Matrix4x4D mat = Mat(4,4,MAT_GRAY_F64);

    if(closeToZeroD(screw.w.length()))
    {
        if(needCalUnit)
        {
            if(closeToZeroD(screw.v.length()))
            {
                return mat;
            }

            mat.setTranslation(screw.v/screw.v.length());
            return mat;
        }
    }

    if(needCalUnit)
    {
        Vector3D omg   = screw.w / screw.w.length();
        Matrix3x3D so3 = SO3D::wedge(omg);

        mat.setRotationMat(so3);
        mat.setTranslation(screw.v);
        return mat;
    }

    Matrix3x3D so3     = SO3D::wedge(screw.w);
    mat.setRotationMat(so3);
    mat.setTranslation(screw.v);
    return mat;
}

ScrewD SE3D::vee(const Matrix4x4D &wed, bool needCalUnit)
{
    Vector3D w = SO3D::vee(wed.getRotationMat());
    Vector3D v = wed.getTranslation();

    if(needCalUnit)
    {
        if(closeToZeroD(w.length()))
        {
            return ScrewD(v/v.length(),Vector3D({0,0,0}));
        }
        else
        {
            return ScrewD(v,w/w.length());
        }
    }

    return ScrewD(v,w);
}

ScrewD SE3D::log()
{

    if((*this) == Mat::eye(4,MAT_GRAY_F64))
    {
        return ScrewD();
    }

    SO3D R         = getRotationMat();
    TranslationD p = getTranslation();

    if(R == Mat::eye(3,MAT_GRAY_F64))
    {          

        return ScrewD(p, Vector3D({0,0,0}));
    }

    R.print();
    Vector3D w        = R.log();

    w.print();

    Matrix3x3D wWedge = SO3D::wedge(w);

    double theta      = w.length();

    std::cout << theta;

    Matrix3x3D GInv   = Mat::eye(3,MAT_GRAY_F64) - 0.5*wWedge + wWedge*wWedge*(1/theta - 0.5*1/tan(0.5*theta))/theta;

    GInv.print();

    Vector3D v        = GInv.mulVec(p);

    return ScrewD(v,w);
}

SE3D SE3D::exp(const ScrewD &screw, double theta)
{
    if(abs(screw.w.length()-1)>MSNH_F64_EPS && abs(screw.w.length())>MSNH_F64_EPS)
    {
        throw Exception(1, "[SE3D] given theta, OMG must be a unit vector ", __FILE__, __LINE__,__FUNCTION__);
    }

    return SE3D::exp(ScrewD(screw.v, screw.w*theta));
}

SE3D SE3D::exp(const ScrewD &screw)
{
    Vector3D v   = screw.v;
    Vector3D omg = screw.w;

    if(closeToZeroD(omg.length()))
    {
        SE3D se3;
        se3.setTranslation(v);
        return se3;
    }
    else
    {
        double theta  = omg.length();

        SO3D so3      = SO3D::exp(omg);

        Vector3D axis = omg/theta;
        v             = v/theta;

        Matrix3x3D wedge = SO3D::wedge(axis);

        Matrix3x3D G    = Mat::eye(3,MAT_GRAY_F64)*theta + (1-cos(theta))*wedge + wedge*wedge*(theta - sin(theta));
        Vector3D p      = G.mulVec(v);

        SE3D se3;
        se3.setRotationMat(so3);
        se3.setTranslation(p);
        return se3;
    }
}

bool SE3D::isSE3(const Mat &mat)
{
    return mat.isHomTransMatrix();
}

}
