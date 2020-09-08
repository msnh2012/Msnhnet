#include "Msnhnet/layers/MsnhEmptyLayer.h"
namespace Msnhnet
{
EmptyLayer::EmptyLayer(const int &batch, const int &width, const int &height, const int &channel)
{
    this->_layerName =  "Empty           ";
    this->_type      =   LayerType::EMPTY;
    this->_batch     =   batch;
    this->_width     =   width;
    this->_height    =   height;
    this->_channel   =   channel;

    this->_outWidth  =   width;
    this->_outHeight =   height;
    this->_outChannel=   channel;
    this->_inputNum  =   width * height * channel;
    this->_outputNum =   this->_outWidth * this->_outHeight * this->_outChannel;

    this->_maxOutputNum  = this->_batch*this->_outputNum;

    char msg[100];
#ifdef WIN32
    sprintf_s(msg, "Empty Layer                  %4d x%4d x%4d -> %4d x%4d x%4d %5.3f BF\n", this->_width, this->_height, this->_channel,
              this->_outWidth, this->_outHeight, this->_outChannel, this->_bFlops);
#else
    sprintf(msg, "Empty Layer                  %4d x%4d x%4d -> %4d x%4d x%4d %5.3f BF\n", this->_width, this->_height, this->_channel,
            this->_outWidth, this->_outHeight, this->_outChannel, this->_bFlops);
#endif
    this->_layerDetail = msg;
}

EmptyLayer::~EmptyLayer()
{
}

void EmptyLayer::forward(NetworkState &netState)
{
    /* Empty layer can't be 0 layer */
    if(this->_layerIndex == 0 && this->_isBranchLayer==false)
    {
        throw Exception(1,"Empty layer should not be 0 layer",__FILE__,__LINE__,__FUNCTION__);
    }

    auto st = TimeUtil::startRecord();

    float* layerInput = netState.getInput();

    if(this->_isBranchLayer) 

    {
        /* 输入/输出 */
        if(this->_isFirstBranch && this->_isLastBranch)

        {
            Blas::cpuCopy(this->_batch*this->_inputNum, netState.input, 1, this->_output, 1); 

        }
        else if(this->_isFirstBranch && !this->_isLastBranch)
        {
            Blas::cpuCopy(this->_batch*this->_inputNum, netState.input, 1, layerInput, 1);

        }
        else if(!this->_isFirstBranch && this->_isLastBranch)
        {
            Blas::cpuCopy(this->_batch*this->_inputNum, layerInput, 1, this->_output, 1);

        }
    }
    else
    {
        /* 输入 */
        if(this->_layerIndex == 0) 

        {
            layerInput      = netState.input;
        }
        else 

        {
            if(netState.net->layers[this->_layerIndex - 1]->getMemReUse() == 0)

            {
                layerInput  = netState.input;
            }
        }
        /* 输出 */
        if(this->_memReUse==1) 

        {

        }
        else

        {
            Blas::cpuCopy(this->_batch*this->_inputNum, layerInput, 1, this->_output, 1);

        }
    }

    this->_forwardTime = TimeUtil::getElapsedTime(st);
}

void EmptyLayer::mallocMemory()
{
    if(!this->_memoryMalloced)
    {
        if(!BaseLayer::isPreviewMode)
        {
            if(!BaseLayer::onlyUseGpu) 

            {
                this->_output     = new float[static_cast<size_t>(this->_outputNum * this->_batch)]();
            }
#ifdef USE_GPU
            if(!BaseLayer::onlyUseCpu)

            {
                this->_gpuOutput  = Cuda::mallocCudaArray(this->_outputNum * this->_batch);
            }
#endif
            this->_memoryMalloced  =  true;
        }
    }
    this->_memReUse         =  0;
}

#ifdef USE_GPU
void EmptyLayer::forwardGPU(NetworkState &netState)
{

    /* Empty layer can't be 0 layer */
    if(this->_layerIndex == 0 && this->_isBranchLayer==false)
    {
        throw Exception(1,"Empty layer should not be 0 layer",__FILE__,__LINE__,__FUNCTION__);
    }

    this->recordCudaStart();

    float* layerGpuInput = netState.getGpuInput();

    if(this->_isBranchLayer) 

    {
        /* 输入/输出 */
        if(this->_isFirstBranch && this->_isLastBranch)

        {
            BlasGPU::gpuCopy(this->_batch*this->_inputNum, netState.input, 1, this->_gpuOutput, 1); 

        }
        else if(this->_isFirstBranch && !this->_isLastBranch)
        {
            BlasGPU::gpuCopy(this->_batch*this->_inputNum, netState.input, 1, layerGpuInput, 1);

        }
        else if(!this->_isFirstBranch && this->_isLastBranch)
        {
            BlasGPU::gpuCopy(this->_batch*this->_inputNum, layerGpuInput, 1, this->_gpuOutput, 1);

        }
    }
    else
    {
        /* 输入 */
        if(this->_layerIndex == 0) 

        {
            layerGpuInput      = netState.input;
        }
        else 

        {
            if(netState.net->layers[this->_layerIndex - 1]->getMemReUse() == 0)

            {
                layerGpuInput  = netState.input;
            }
        }
        /* 输出 */
        if(this->_memReUse==1) 

        {

        }
        else

        {
            BlasGPU::gpuCopy(this->_batch*this->_inputNum, layerGpuInput, 1, this->_gpuOutput, 1);

        }
    }

    this->recordCudaStop();
}
#endif
}
