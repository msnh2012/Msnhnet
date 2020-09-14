#include "Msnhnet/layers/MsnhPaddingLayer.h"
namespace Msnhnet
{

PaddingLayer::PaddingLayer(const int &batch, const int &height, const int &width, const int &channel, const int &top,
                           const int &down, const int &left, const int &right, const float &paddingVal)
{
    this->_type          =   LayerType::PADDING;
    this->_layerName     =   "Padding         ";
    this->_batch         =   batch;
    this->_height        =   height;
    this->_width         =   width;
    this->_channel       =   channel;

    this->_inputNum      =   width * height * channel;

    this->_top           =   top;
    this->_down          =   down;
    this->_left          =   left;
    this->_right         =   right;

    this->_paddingVal    =   paddingVal;

    this->_outHeight     =   this->_height + this->_top + this->_down;
    this->_outWidth      =   this->_width   + this->_left + this->_right;
    this->_outChannel    =   this->_channel;

    this->_outputNum     =   this->_outHeight * this->_outWidth * this->_outChannel;

    char msg[100];

    this->_maxOutputNum  = this->_batch*this->_outputNum;

#ifdef WIN32
    sprintf_s(msg, "Padding                      %4d x%4d x%4d -> %4d x%4d x%4d %5.3f BF\n", this->_width, this->_height, this->_channel,
              this->_outWidth, this->_outHeight, this->_outChannel, this->_bFlops);
#else
    sprintf(msg, "Padding                      %4d x%4d x%4d -> %4d x%4d x%4d %5.3f BF\n", this->_width, this->_height, this->_channel,
            this->_outWidth, this->_outHeight, this->_outChannel, this->_bFlops);
#endif
    this->_layerDetail = msg;
}

void PaddingLayer::mallocMemory()
{
    if(!this->_memoryMalloced)
    {
        if(!BaseLayer::isPreviewMode)
        {
            if(!BaseLayer::onlyUseGpu) 

            {
                this->_output        =   new float[static_cast<size_t>(this->_outputNum * this->_batch)]();
            }
#ifdef USE_GPU
            if(!BaseLayer::onlyUseCpu)

            {
                this->_gpuOutput     =   Cuda::mallocCudaArray(this->_outputNum * this->_batch);
            }
#endif
            this->_memoryMalloced  =  true;
        }
    }
    this->_memReUse         =  0;
}

void PaddingLayer::forward(NetworkState &netState)
{
    auto st = TimeUtil::startRecord();

    float* layerInput   = netState.getInput();
    float* layerOutput  = nullptr;

    /* 输入 */
    if(this->_isBranchLayer) 

    {
        if(this->_isFirstBranch)

        {
            layerInput      = netState.input;
        }
    }
    else
    {
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
    }

    /* 输出 */
    if(this->_isBranchLayer) 

    {
        if(this->_isLastBranch)

        {
            layerOutput     = this->_output; 

        }
        else 

        {
            layerOutput     = netState.getOutput(); 

            netState.shuffleInOut();

        }
    }
    else
    {
        if(this->_memReUse==1) 

        {
            layerOutput     = netState.getOutput(); 

            netState.shuffleInOut();

        }
        else

        {
            layerOutput     = this->_output;
        }
    }

#ifdef USE_ARM1/*TODO*/
    PaddingLayerArm::padding(layerInput, this->_width, this->_height, this->_channel, layerOutput, this->_top, this->_down, this->_left, this->_right, this->_paddingVal);
#else
    for (int i = 0; i < this->_batch; ++i)
    {
#ifdef USE_OMP
#pragma omp parallel for num_threads(OMP_THREAD)
#endif
        for (int j = 0; j < this->_outChannel; ++j)
        {
            for (int m = 0; m < this->_outHeight; ++m)
            {
                for (int n = 0; n < this->_outWidth; ++n)
                {
                    float val = 0;

                    if(m < this->_top || (m >= (this->_height + this->_top)))
                    {
                        val     =   this->_paddingVal;
                    }
                    else
                    {
                        if(n < this->_left || (n >= (this->_width + this->_left)))
                        {
                            val     =   this->_paddingVal;
                        }
                        else
                        {
                            val     =   layerInput[ i*this->_channel*this->_height*this->_width + j*this->_height*this->_width + (m-this->_top)*this->_width + (n - this->_left)];
                        }
                    }

                    layerOutput[i*this->_outChannel*this->_outHeight*this->_outWidth + j*this->_outHeight*this->_outWidth + m*this->_outWidth + n] = val;

                }
            }
        }
    }
#endif

    this->_forwardTime = TimeUtil::getElapsedTime(st);

}

#ifdef USE_GPU
void PaddingLayer::forwardGPU(NetworkState &netState)
{
    this->recordCudaStart();

    float* layerGpuInput   = netState.getGpuInput();
    float* layerGpuOutput  = nullptr;

    /* 输入 */
    if(this->_isBranchLayer) 

    {
        if(this->_isFirstBranch)

        {
            layerGpuInput      = netState.input;
        }
    }
    else
    {
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
    }

    /* 输出 */
    if(this->_isBranchLayer) 

    {
        if(this->_isLastBranch)

        {
            layerGpuOutput     = this->_gpuOutput; 

        }
        else 

        {
            layerGpuOutput     = netState.getGpuOutput(); 

            netState.shuffleGpuInOut();

        }
    }
    else
    {
        if(this->_memReUse==1) 

        {
            layerGpuOutput     = netState.getGpuOutput(); 

            netState.shuffleGpuInOut();

        }
        else

        {
            layerGpuOutput     = this->_gpuOutput;
        }
    }

    PaddingLayerGPU::forwardNormalGPU(this->_batch, this->_outChannel, this->_outHeight, this->_outWidth,
                                      this->_height, this->_width, this->_channel,
                                      this->_top, this->_left,
                                      this->_paddingVal,
                                      layerGpuInput,
                                      layerGpuOutput
                                      );
    this->recordCudaStop();
}
#endif

int PaddingLayer::getTop() const
{
    return _top;
}

int PaddingLayer::getDown() const
{
    return _down;
}

int PaddingLayer::getLeft() const
{
    return _left;
}

int PaddingLayer::getRight() const
{
    return _right;
}

float PaddingLayer::getPaddingVal() const
{
    return _paddingVal;
}

}
