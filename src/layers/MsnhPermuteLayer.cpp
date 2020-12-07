#include "Msnhnet/layers/MsnhPermuteLayer.h"
namespace Msnhnet
{

PermuteLayer::PermuteLayer(const int &batch, const int &height, const int &width, const int &channel, const int &dim0, const int &dim1, const int &dim2)
{
    this->_batch    =   batch;
    this->_channel  =   channel;
    this->_height   =   height;
    this->_width    =   width;

    this->_layerName =  "Permute         ";
    this->_type      =   LayerType::PERMUTE;

    if(dim0 != 0 && dim0 != 1 && dim0 !=2 &&
            dim1 != 0 && dim1 != 1 && dim1 !=2 &&
            dim2 != 0 && dim2 != 1 && dim2 !=2)
    {
        throw Exception(1,"dim value must be 0/1/2 --> c/h/w",__FILE__, __LINE__, __FUNCTION__);
    }

    if(dim0 == dim1 || dim1 == dim2 || dim2 == dim0)
    {
        throw Exception(1,"dims can't equal",__FILE__, __LINE__, __FUNCTION__);
    }

    this->_dim0 = dim0;
    this->_dim1 = dim1;
    this->_dim2 = dim2;

    if(dim0 == 0)
    {
        this->_outChannel = channel;
    }
    else if(dim0 == 1)
    {
        this->_outChannel = height;
    }
    else if(dim0 == 2)
    {
        this->_outChannel = width;
    }

    if(dim1 == 0)
    {
        this->_outHeight = channel;
    }
    else if(dim1 == 1)
    {
        this->_outHeight = height;
    }
    else if(dim1 == 2)
    {
        this->_outHeight = width;
    }

    if(dim2 == 0)
    {
        this->_outWidth = channel;
    }
    else if(dim2 == 1)
    {
        this->_outWidth = height;
    }
    else if(dim2 == 2)
    {
        this->_outWidth = width;
    }

    this->_inputNum  =   width * height * channel;
    this->_outputNum =   this->_outWidth * this->_outHeight * this->_outChannel;

    this->_maxOutputNum  = this->_batch*this->_outputNum;

    char msg[100];
#ifdef WIN32
    sprintf_s(msg, "Permute                      %4d x%4d x%4d -> %4d x%4d x%4d %5.3f BF\n", this->_width, this->_height, this->_channel,
              this->_outWidth, this->_outHeight, this->_outChannel, this->_bFlops);
#else
    sprintf(msg, "Permute                      %4d x%4d x%4d -> %4d x%4d x%4d %5.3f BF\n", this->_width, this->_height, this->_channel,
            this->_outWidth, this->_outHeight, this->_outChannel, this->_bFlops);
#endif
    this->_layerDetail = msg;
}

PermuteLayer::~PermuteLayer()
{

}

void PermuteLayer::forward(NetworkState &netState)
{
    auto st = TimeUtil::startRecord();

    float* layerInput   = netState.getInput();
    float* layerOutput  = nullptr;

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

    if(this->_memReUse==1) 

    {
        layerOutput     = netState.getOutput(); 

        netState.shuffleInOut();

    }
    else

    {
        layerOutput     = this->_output;
    }

    for (int b = 0; b < this->_batch; ++b)
    {
#ifdef USE_OMP
#pragma omp parallel for num_threads(OMP_THREAD)
#endif
        for (int c = 0; c < this->_outChannel; ++c)
        {
            for (int h = 0; h < this->_outHeight; ++h)
            {
                for (int w = 0; w < this->_outWidth; ++w)
                {
                    int cc = 0;
                    int hh = 0;
                    int ww = 0;
                    if(this->_dim0 == 0 && this->_dim1 == 1 &&this->_dim2 == 2)
                    {
                        cc = c;
                        hh = h;
                        ww = w;
                    }
                    else if(this->_dim0 == 0 && this->_dim1 == 2 &&this->_dim2 == 1)
                    {
                        cc = c;
                        hh = w;
                        ww = h;
                    }
                    else if(this->_dim0 == 1 && this->_dim1 == 0 &&this->_dim2 == 2)
                    {
                        cc = h;
                        hh = c;
                        ww = w;
                    }
                    else if(this->_dim0 == 1 && this->_dim1 == 2 &&this->_dim2 == 0)
                    {
                        cc = w;
                        hh = c;
                        ww = h;
                    }
                    else if(this->_dim0 == 2 && this->_dim1 == 0 &&this->_dim2 == 1)
                    {
                        cc = h;
                        hh = w;
                        ww = c;
                    }
                    else if(this->_dim0 == 2 && this->_dim1 == 1 &&this->_dim2 == 0)
                    {
                        cc = w;
                        hh = h;
                        ww = c;
                    }

                    layerOutput[b*this->_outChannel*this->_outWidth*this->_outHeight + c*this->_outWidth*this->_outHeight + h*this->_outWidth + w]
                            =
                    layerInput[b*this->_channel*this->_width*this->_height + cc*this->_width*this->_height + hh*this->_width + ww];
                }
            }
        }
    }

    this->_forwardTime = TimeUtil::getElapsedTime(st);
    return;
}

void PermuteLayer::mallocMemory()
{
    if(!this->_memoryMalloced)
    {
        if(!BaseLayer::isPreviewMode)
        {
            if(!BaseLayer::onlyUseGpu) 

            {

                this->_output         = MemoryManager::effcientNew<float>(static_cast<size_t>(this->_outputNum * this->_batch));
            }
#ifdef USE_GPU
            if(!BaseLayer::onlyUseCpu)

            {
                this->_gpuOutput =   Cuda::mallocCudaArray(this->_outputNum * this->_batch);
            }
#endif
            this->_memoryMalloced  =  true;
        }
    }
    this->_memReUse         =  0;
}

#ifdef USE_GPU
void PermuteLayer::forwardGPU(NetworkState &netState)
{
    this->recordCudaStart();

    float* layerGpuInput   = netState.getGpuOutput();
    float* layerGpuOutput  = nullptr;

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

    if(this->_memReUse==1) 

    {
        layerGpuOutput     = netState.getGpuOutput(); 

        netState.shuffleGpuInOut();

    }
    else

    {
        layerGpuOutput     = this->_gpuOutput;
    }

    PermuteLayerGPU::forwardNormalGPU(this->_batch, this->_outChannel, this->_outHeight, this->_outWidth,
                                      this->_height, this->_width, this->_channel,
                                      this->_dim0, this->_dim1, this->_dim2,
                                      layerGpuInput, layerGpuOutput
                                      );
    this->recordCudaStop();
}
#endif

int PermuteLayer::getDim0() const
{
    return _dim0;
}

int PermuteLayer::getDim1() const
{
    return _dim1;
}

int PermuteLayer::getDim2() const
{
    return _dim2;
}

}
