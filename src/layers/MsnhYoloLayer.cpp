#include "Msnhnet/layers/MsnhYoloLayer.h"

namespace Msnhnet
{
YoloLayer::YoloLayer(const int &batch, const int &width, const int &height, const int &num, const int &orgWidth, const int &orgHeight, const int &classNum,
                     const std::vector<float> &anchors, const YoloType &yoloType)
{
    this->_type      =   LayerType::YOLO;
    this->_layerName =   "Yolo            ";
    this->_num       =   num;
    this->_batch     =   batch;
    this->_height    =   height;
    this->_width     =   width;
    this->_channel   =   num;

    this->_orgHeight =   orgHeight;
    this->_orgWidth  =   orgWidth;

    this->_classNum  =   classNum;

    this->_outWidth  =   this->_width;
    this->_outHeight =   this->_height;
    this->_outChannel=   this->_channel;

    this->_outputNum =   this->_height*this->_width*this->_num;
    this->_inputNum  =   this->_outputNum;

    this->_maxOutputNum  = this->_batch*this->_outputNum;

    this->anchors   =   anchors;

    this->_yoloType =   yoloType;

    this->_ratios    =   1.f*orgHeight/_outHeight;  

    if(3*(this->_classNum + 4 + 1) != num)
    {
        throw Exception(1, "class num error!", __FILE__, __LINE__, __FUNCTION__);
    }

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
    }
    this->_memReUse   =   0;
    this->_layerDetail.append("Yolo. class num : " + std::to_string(classNum) + "\n");
}

void YoloLayer::forward(NetworkState &netState)
{
    /*Yolo layer should not be 0 layer */
    if(this->_layerIndex == 0)
    {
        throw Exception(1,"Yolo layer should not be 0 layer",__FILE__,__LINE__,__FUNCTION__);
    }

    /*Yolo layer should not be 0 layer */
    if(this->_isBranchLayer)
    {
        throw Exception(1,"Yolo layer should not be branch layer",__FILE__,__LINE__,__FUNCTION__);
    }

    auto st = TimeUtil::startRecord();
    float* layerInput   = netState.getInput();

    if(netState.net->layers[this->_layerIndex - 1]->getMemReUse() == 0)

    {
        layerInput  = netState.input;
    }

    Blas::cpuCopy(netState.inputNum, layerInput, 1, this->_output, 1);

    if(static_cast<int>(this->_yoloType)/10 == 3 || static_cast<int>(this->_yoloType)/10 == 4)
    {
        for (int b = 0; b < this->_batch; ++b)
        {
            for (int n = 0; n < 3; ++n)
            {

                int index = b*this->_outputNum + n*this->_width*this->_height*( 4 + 1 + this->_classNum);
                exSigmoid(this->_output + index, this->_width, this->_height, this->_ratios, true);

                index = index + this->_width*this->_height;

                exSigmoid(this->_output + index, this->_width, this->_height, this->_ratios, false);

                index = index + this->_width*this->_height;

                aExpT(this->_output + index, this->_width*this->_height, anchors[n%3*2]);

                index = index + this->_width*this->_height;
                aExpT(this->_output + index, this->_width*this->_height, anchors[n%3*2 + 1]);

                index = index + this->_width*this->_height;
                sigmoid(this->_output + index, this->_width*this->_height*(1+this->_classNum));
            }

        }
    }
    else if(static_cast<int>(this->_yoloType)/10 == 5)
    {
        for (int b = 0; b < this->_batch; ++b)
        {
            for (int n = 0; n < 3; ++n)
            {

                int index = b*this->_outputNum + n*this->_width*this->_height*( 4 + 1 + this->_classNum);
                exSigmoidV5(this->_output + index, this->_width, this->_height, this->_ratios, true);

                index = index + this->_width*this->_height;

                exSigmoidV5(this->_output + index, this->_width, this->_height, this->_ratios, false);

                index = index + this->_width*this->_height;

                aPowSigmoid(this->_output + index, this->_width*this->_height, anchors[n%3*2]);

                index = index + this->_width*this->_height;
                aPowSigmoid(this->_output + index, this->_width*this->_height, anchors[n%3*2 + 1]);

                index = index + this->_width*this->_height;
                sigmoid(this->_output + index, this->_width*this->_height*(1+this->_classNum));
            }

        }
    }

    this->_forwardTime =   TimeUtil::getElapsedTime(st);
}

#ifdef USE_GPU
void YoloLayer::forwardGPU(NetworkState &netState)
{
    this->recordCudaStart();

    float* layerGpuInput   = netState.getGpuInput();

    if(netState.net->layers[this->_layerIndex - 1]->getMemReUse() == 0)

    {
        layerGpuInput  = netState.input;
    }

    BlasGPU::gpuCopy(netState.inputNum, layerGpuInput, 1, this->_gpuOutput, 1);

    int num     = this->_width*this->_height;
    int nxClass =  num * (1+this->_classNum);

    if(static_cast<int>(this->_yoloType)/10 == 3 || static_cast<int>(this->_yoloType)/10 == 4)
    {
        for (int b = 0; b < this->_batch; ++b)
        {
            for (int n = 0; n < 3; ++n)
            {

                int index = b*this->_outputNum + n*this->_width*this->_height*( 4 + 1 + this->_classNum);
                YoloLayerGPU::exSigmoidGpu(num, this->_gpuOutput + index, this->_width, this->_ratios, 1);

                index = index + this->_width*this->_height;
                YoloLayerGPU::exSigmoidGpu(num, this->_gpuOutput + index, this->_width, this->_ratios, 0);

                index = index + this->_width*this->_height;
                YoloLayerGPU::aExpTGpu(num, this->_gpuOutput + index, anchors[n%3*2]);

                index = index + this->_width*this->_height;
                YoloLayerGPU::aExpTGpu(num, this->_gpuOutput + index, anchors[n%3*2 + 1]);

                index = index + this->_width*this->_height;
                YoloLayerGPU::sigmoidGpu(nxClass, this->_gpuOutput + index);
            }

        }
    }
    else if( static_cast<int>(this->_yoloType)/10 == 5)
    {
        for (int b = 0; b < this->_batch; ++b)
        {
            for (int n = 0; n < 3; ++n)
            {

                int index = b*this->_outputNum + n*this->_width*this->_height*( 4 + 1 + this->_classNum);
                YoloLayerGPU::exSigmoidV5Gpu(num, this->_gpuOutput + index, this->_width, this->_ratios, 1);

                index = index + this->_width*this->_height;
                YoloLayerGPU::exSigmoidV5Gpu(num, this->_gpuOutput + index, this->_width, this->_ratios, 0);

                index = index + this->_width*this->_height;
                YoloLayerGPU::aPowSigmoid(num, this->_gpuOutput + index, anchors[n%3*2]);

                index = index + this->_width*this->_height;
                YoloLayerGPU::aPowSigmoid(num, this->_gpuOutput + index, anchors[n%3*2 + 1]);

                index = index + this->_width*this->_height;
                YoloLayerGPU::sigmoidGpu(nxClass, this->_gpuOutput + index);
            }

        }
    }

    this->recordCudaStop();
}
#endif

void YoloLayer::sigmoid(float *val, const int &num)
{
#ifdef USE_OMP
#pragma omp parallel for num_threads(OMP_THREAD)
#endif
    for (int i = 0; i < num; ++i)
    {
        val[i] = 1.f/(1.f+expf(-val[i]));
    }
}

void YoloLayer::exSigmoid(float *val, const int &width, const int&height, const float &ratios, const bool &addGridW)
{
#ifdef USE_OMP
#pragma omp parallel for num_threads(OMP_THREAD)
#endif
    for (int i = 0; i < width*height; ++i)
    {
        if(addGridW)
        {
            val[i] = (1.f/(1.f+expf(-val[i])) + i%width)*ratios;
        }
        else
        {
            val[i] = (1.f/(1.f+expf(-val[i])) + i/width)*ratios;
        }
    }
}

void YoloLayer::exSigmoidV5(float *val, const int &width, const int &height, const float &ratios, const bool &addGridW)
{
#ifdef USE_OMP
#pragma omp parallel for num_threads(OMP_THREAD)
#endif
    for (int i = 0; i < width*height; ++i)
    {
        if(addGridW)
        {
            val[i] = (2.f/(1.f+expf(-val[i])) - 0.5f + i%width)*ratios;
        }
        else
        {
            val[i] = (2.f/(1.f+expf(-val[i])) - 0.5f + i/width)*ratios;
        }
    }
}

void YoloLayer::aExpT(float *val, const int &num, const float &a)
{
#ifdef USE_OMP
#pragma omp parallel for num_threads(OMP_THREAD)
#endif
    for (int i = 0; i < num; ++i)
    {
        val[i] = a*expf(val[i]);
    }
}

void YoloLayer::aPowSigmoid(float *val, const int &num, const float &a)
{
#ifdef USE_OMP
#pragma omp parallel for num_threads(OMP_THREAD)
#endif
    for (int i = 0; i < num; ++i)
    {
        val[i] = a*powf(2.f/(1.f+expf(-val[i])),2.f);
    }
}

int YoloLayer::getClassNum() const
{
    return _classNum;
}

int YoloLayer::getOrgHeight() const
{
    return _orgHeight;
}

int YoloLayer::getOrgWidth() const
{
    return _orgWidth;
}

float YoloLayer::getRatios() const
{
    return _ratios;
}

YoloType YoloLayer::getYoloType() const
{
    return _yoloType;
}
}
