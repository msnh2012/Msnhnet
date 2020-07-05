#ifndef MSNHINFERENCECFG_H
#define MSNHINFERENCECFG_H
#include <stdint.h>
#include <float.h>
#include <string>
#include <vector>
#include <chrono>
#include "Msnhnet/utils/MsnhException.h"
#include "Msnhnet/config/MsnhnetMacro.h"

#ifdef USE_OMP
#include <omp.h>
#endif

#ifdef USE_ARM
#define USE_NEON
#endif

#ifdef USE_NEON
#include <arm_neon.h>
#endif

#ifdef USE_OPEN_BLAS
#include <cblas.h>
#endif

#ifndef OMP_THREAD
#define OMP_THREAD omp_get_max_threads()
#endif

enum ActivationType
{
    LOGISTIC,
    RELU,
    RELU6,
    RELIE,
    LINEAR,
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
    SOFTMAX,
    DETECTION,
    CROP,
    ROUTE,
    NORMALIZATION,
    AVGPOOL,
    LOCAL,
    SHORTCUT,
    SCALE_CHANNELS,
    SAM,
    ACTIVE,
    RNN,
    GRU,
    LSTM,
    CONV_LSTM,
    CRNN,
    BATCHNORM,
    NETWORK,
    XNOR,
    REGION,
    YOLOV3,
    YOLOV3_OUT,
    GAUSSIAN_YOLO,
    ISEG,
    REORG,
    REORG_OLD,
    UPSAMPLE,
    LOGXENT,
    L2NORM,
    EMPTY,
    BLANK,
    CONFIG,
    RES_BLOCK,
    RES_2_BLOCK,
    CONCAT_BLOCK,
    ADD_BLOCK,
    PADDING
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

#endif // MSNHINFERENCECFG_H
