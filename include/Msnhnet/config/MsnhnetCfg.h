#ifndef MSNHINFERENCECFG_H
#define MSNHINFERENCECFG_H
#include <stdint.h>
#include <float.h>
#include <string>
#include <vector>
#include <chrono>
#include "Msnhnet/utils/MsnhException.h"
#include "Msnhnet/config/MsnhnetMacro.h"
#include <math.h>
#include <string.h>

#ifdef USE_OMP
#include <omp.h>
#endif

#ifdef USE_NEON
#include <arm_neon.h>
#endif

#ifdef USE_OPEN_BLAS
#include <cblas.h>
#endif

#ifdef USE_NNPACK
#include <nnpack.h>
#endif

#ifndef OMP_THREAD
#define OMP_THREAD omp_get_max_threads()
#endif

#define CUDA_THREADS 512

#define MIN_OMP_DATA 10000

#define MSNHNET_VERSION 1100

enum ActivationType
{
    LOGISTIC,
    RELU,
    RELU6,
    RELIE,
    RAMP,
    TANH,
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
    CONVOLUTIONAL,
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
    SLICE,
    REDUCTION,
    CONFIG,
    RES_BLOCK,
    RES_2_BLOCK,
    CONCAT_BLOCK,
    ADD_BLOCK,
    PADDING
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
    REDUCTION_SUM,
    REDUCTION_MEAN
};

enum WeightsType
{
    NO_WEIGHTS,
    PER_FEATURE,
    PER_CHANNEL
};

enum WeightsNorm
{
    NO_NORM,
    RELU_NORM,
    SOFTMAX_NORM
};

#endif 

