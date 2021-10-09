#ifndef MSNHINFERENCECFG_H
#define MSNHINFERENCECFG_H
#include <stdint.h>
#include <float.h>
#include <string>
#include <vector>
#include <chrono>
#include "Msnhnet/utils/MsnhException.h"
#include <string.h>
#include <assert.h>

#ifdef __WIN32__
#include <math.h>
#endif

#ifdef __linux__
#include <cmath>
#endif

#ifdef USE_OMP
#include <omp.h>
#endif

#ifdef USE_NEON
#include <arm_neon.h>
#endif

#ifdef USE_OPEN_BLAS
#include <cblas.h>
#endif

#ifdef USE_OPENGL
#include <Msnhnet/config/MsnhnetOpenGL.h>
#endif

#ifndef OMP_THREAD
#define OMP_THREAD omp_get_max_threads()
#endif

#define CUDA_THREADS 512

#define MIN_OMP_DATA 10000

#define MSNHNET_VERSION 2000

#define EFFCIENT_ALIGN 16

#define MSNH_F32_EPS 1E-6

#define MSNH_F64_EPS 1E-12

#define MSNH_2_PI 6.28318530906

#define MSNH_PI   3.14159265453

#define MSNH_PI_2 1.57079632726

#define MSNH_PI_3 1.04719755151

#define MSNH_PI_4 0.78539816363

#define MSNH_PI_6 0.52359877575

#define MSNH_RAD_2_DEG 57.295779495935

#define MSNH_DEG_2_RAD 0.0174532925251

#define clip(x,a,b) ((x<a)?a:(x>b)?b:x)

#define m_swap(a,b) (a=(a)+(b),b=(a)-(b),a=(a)-(b))

#define deg2radf(deg) ((float)(deg/180.f*MSNH_PI))

#define deg2radd(deg) ((double)(deg/180.0*MSNH_PI))

#define rad2degf(deg) ((float)(deg/MSNH_PI*180.f))

#define rad2degd(deg) ((double)(deg/MSNH_PI*180.0))

#define closeToZeroD(x) (fabs(x)<MSNH_F64_EPS)

#define closeToZeroF(x) (fabsf(x)<MSNH_F32_EPS)

#ifndef M_PI
#define M_PI 3.14159265453
#endif

#ifndef ROT_EPS
#define ROT_EPS 0.00001
#endif

#define USE_R_VALUE_REF 1
namespace Msnhnet
{
enum ActivationType
{
    LOGISTIC    =   0,
    RELU,
    RELU6,
    RELIE,
    RAMP,
    TANH,
    PRELU,
    PLSE,
    LEAKY,
    ELU,
    LOGGY,
    STAIR,
    HARDTAN,
    LHTAN,
    SOFT_PLUS,
    SELU,
    SWISH,
    HARD_SWISH,
    MISH,
    NORM_CHAN,
    NORM_CHAN_SOFTMAX,
    NORM_CHAN_SOFTMAX_MAXVAL,
    NONE
};

enum LayerType
{
    CONVOLUTIONAL   =   0,
    DECONVOLUTIONAL,
    CONNECTED,
    MAXPOOL,
    LOCAL_AVGPOOL,
    GLOBAL_AVGPOOL,
    SOFTMAX,
    CROP,
    ROUTE,
    VARIABLE_OP,
    NORMALIZATION,
    AVGPOOL,
    ACTIVE,
    BATCHNORM,
    NETWORK,
    YOLO,
    YOLO_OUT,
    GAUSSIAN_YOLO,
    UPSAMPLE,
    L2NORM,
    EMPTY,
    VIEW,
    PERMUTE,
    PIXEL_SHUFFLE,
    SLICE,
    REDUCTION,
    CONFIG,
    RES_BLOCK,
    RES_2_BLOCK,
    CONCAT_BLOCK,
    ADD_BLOCK,
    PADDING,
    CLIP
};

enum Arithmetic
{
    ARITH_ADD = 0,
    ARITH_SUB,
    ARITH_SUB_INV,
    ARITH_MUL,
    ARITH_DIV,
    ARITH_DIV_INV
};

enum Scientific
{
    SCI_ABS=0,
    SCI_ACOS,
    SCI_ASIN,
    SCI_ATAN,
    SCI_COS,
    SCI_COSH,
    SCI_SIN,
    SCI_SINH,
    SCI_TAN,
    SCI_TANH,
    SCI_EXP,
    SCI_POW,
    SCI_LOG,
    SCI_LOG10,
    SCI_SQRT
};

enum ReductionType
{
    REDUCTION_SUM   =   0,
    REDUCTION_MEAN
};

enum WeightsType
{
    NO_WEIGHTS  =   0,
    PER_FEATURE,
    PER_CHANNEL
};

enum WeightsNorm
{
    NO_NORM =   0,
    RELU_NORM,
    SOFTMAX_NORM
};

enum RotSequence
{
    ROT_XYZ,
    ROT_XZY,
    ROT_YXZ,
    ROT_YZX,
    ROT_ZXY,
    ROT_ZYX,
};
}

#endif 

