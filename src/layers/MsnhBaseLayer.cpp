#include "Msnhnet/layers/MsnhBaseLayer.h"

namespace Msnhnet
{

bool BaseLayer::supportAvx      = false;
bool BaseLayer::supportFma      = false;
bool BaseLayer::isPreviewMode   = false;
bool BaseLayer::onlyUseCuda     = false;
bool BaseLayer::onlyUseCpu      = false;
bool BaseLayer::onlyUseGpu      = false;

bool BaseLayer::useFp16         = false;

#ifdef USE_GPU
cudaEvent_t  BaseLayer::_start;
cudaEvent_t  BaseLayer::_stop;
#endif

void BaseLayer::initSimd()
{
    std::cout<<"Msnhnet Version : "<<MSNHNET_VERSION/1000<<"."<<MSNHNET_VERSION%1000<<std::endl;
    std::cout<<"Checking......"<<std::endl;
#ifdef USE_X86
    SimdInfo info;
    info.checkSimd();

    supportAvx = info.getSupportAVX2();
    supportFma = info.getSupportFMA3();

    if(supportAvx&&supportFma)
    {
        std::cout<<"avx2 speed up"<<std::endl;
    }
#endif

#if USE_ARM
#ifdef USE_NEON
    std::cout<<"Use NEON."<<std::endl;
#endif

#ifdef USE_NNPACK
    std::cout<<"Use NNPACK."<<std::endl;
#endif
#endif

#ifdef USE_OMP
    std::cout<<"Use OMP.\nOMP thread num : "<<OMP_THREAD <<std::endl;
#endif

#ifdef USE_GPU
    std::cout<<Cuda::getDeviceInfo()<<std::endl;
#ifdef USE_CUDNN
    std::cout<<"Use CUDNN"<<std::endl;
    std::cout<<"CUDNN version: "<<cudnnGetVersion()/1000<<"."<< (cudnnGetVersion() - cudnnGetVersion()/1000*1000)/100 <<std::endl;

    if(cudnnGetVersion() < 7000)
    {
        throw Exception(1,"cudnn version must > 7.0",__FILE__,__LINE__,__FUNCTION__);
    }
#endif
#endif
}

LayerType BaseLayer::type() const
{
    return _type;
}

ActivationType BaseLayer::activation() const
{
    return _activation;
}

int BaseLayer::getOutHeight() const
{
    return _outHeight;
}

int BaseLayer::getOutWidth() const
{
    return _outWidth;
}

int BaseLayer::getOutChannel() const
{
    return _outChannel;
}

int BaseLayer::getOutputNum() const
{
    return _outputNum;
}

void BaseLayer::setOutHeight(int outHeight)
{
    _outHeight = outHeight;
}

void BaseLayer::setOutWidth(int outWidth)
{
    _outWidth = outWidth;
}

void BaseLayer::setOutChannel(int outChannel)
{
    _outChannel = outChannel;
}

float *BaseLayer::getOutput() const
{
    return _output;
}

int BaseLayer::getInputNum() const
{
    return _inputNum;
}

size_t BaseLayer::getWorkSpaceSize() const
{
    return _workSpaceSize;
}

void BaseLayer::setWorkSpaceSize(const size_t &workSpaceSize)
{
    _workSpaceSize = workSpaceSize;
}

size_t BaseLayer::getNumWeights() const
{
    return _numWeights;
}

std::string BaseLayer::getLayerDetail() const
{
    return _layerDetail;
}

int BaseLayer::getHeight() const
{
    return _height;
}

int BaseLayer::getWidth() const
{
    return _width;
}

int BaseLayer::getChannel() const
{
    return _channel;
}

float BaseLayer::getForwardTime() const
{
    return _forwardTime;
}

std::string BaseLayer::getLayerName() const
{
    return _layerName;
}

int BaseLayer::getBatch() const
{
    return _batch;
}

ActivationType BaseLayer::getActivation() const
{
    return _activation;
}

size_t BaseLayer::getInputSpaceSize() const
{
    return _inputSpaceSize;
}

uint8_t BaseLayer::getMemReUse() const
{
    return _memReUse;
}

int BaseLayer::getLayerIndex() const
{
    return _layerIndex;
}

void BaseLayer::setLayerIndex(int layerIndex)
{
    _layerIndex = layerIndex;
}

size_t BaseLayer::getMaxOutputNum() const
{
    return _maxOutputNum;
}

void BaseLayer::setIsBranchLayer(bool isBranchLayer)
{
    _isBranchLayer = isBranchLayer;
}

void BaseLayer::setBranchFirst(bool branchFirst)
{
    _isFirstBranch = branchFirst;
}

void BaseLayer::setBranchLast(bool branchLast)
{
    _isLastBranch = branchLast;
}

BaseLayer::BaseLayer()
{

}

BaseLayer::~BaseLayer()
{
    releaseArr(_output);
#ifdef USE_GPU
    Cuda::freeCuda(_gpuOutput);

#endif
}

void BaseLayer::setPreviewMode(const bool &previewMode)
{
    BaseLayer::isPreviewMode = previewMode;
}

#ifdef USE_GPU
void BaseLayer::setForceUseCuda(const bool &forceUseCuda)
{
    BaseLayer::onlyUseCuda = forceUseCuda;
}

void BaseLayer::setUseFp16(const bool &useFp16)
{
    BaseLayer::useFp16 = useFp16;
}

void BaseLayer::setOnlyGpu(const bool &onlyGpu)
{
    BaseLayer::onlyUseGpu = onlyGpu;
}

void BaseLayer::setOnlyCpu(const bool &onlyCpu)
{
    BaseLayer::onlyUseCpu = onlyCpu;
}
#endif

void BaseLayer::mallocMemory()
{

}

void BaseLayer::forward(NetworkState &netState)
{
    (void)netState;
}

void BaseLayer::loadAllWeigths(std::vector<float> &weights)
{
    (void)weights;
}

void BaseLayer::saveAllWeights(const int &mainIdx, const int &branchIdx, const int &branchIdx1)
{
    (void)mainIdx;
    (void)branchIdx;
    (void)branchIdx1;
}

#ifdef USE_GPU
std::vector<float> BaseLayer::getVecFromCuda(float * const data, const int &num)
{
    float* out = new float[num]();
    Cuda::pullCudaArray(data, out,num);
    std::vector<float> datVec{out,out+num};
    delete[] out;
    out = nullptr;
    return datVec;
}

void BaseLayer::forwardGPU(NetworkState &netState)
{
    (void)netState;
}

float *BaseLayer::getGpuOutput() const
{
    return _gpuOutput;
}

void BaseLayer::recordCudaStart()
{
    /* cudaEventRecord(_start, 0); */
}

void BaseLayer::recordCudaStop()
{

    /* Not good */

    this->_forwardTime = 0.f;
    /*
   cudaThreadSynchronize();
   cudaEventRecord(_stop, 0);
   cudaEventSynchronize(_stop);
   float elapsedTime;
   cudaEventElapsedTime(&elapsedTime, _start, _stop);
   this->_forwardTime = elapsedTime;
*/
}
#endif
}
