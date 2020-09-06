#include "Msnhnet/layers/MsnhYolov3Layer.h"

namespace Msnhnet
{
Yolov3Layer::Yolov3Layer(const int &batch, const int &width, const int &height, const int &num, const int &orgWidth, const int &orgHeight, const int &classNum, const std::vector<float> &anchors)
{
    this->_type      =   LayerType::YOLOV3;
    this->_layerName =   "Yolov3          ";
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

    this->anchors   =   anchors;

    this->_ratios    =   1.f*orgHeight/_outHeight;  

    if(3*(this->_classNum + 4 + 1) != num)
    {
        throw Exception(1, "class num error!", __FILE__, __LINE__, __FUNCTION__);
    }

    if(!BaseLayer::isPreviewMode)
    {
        this->_output    =   new float[static_cast<size_t>(this->_outputNum * this->_batch)]();
#ifdef USE_GPU
        this->_gpuOutput =   Cuda::makeCudaArray(this->_output, this->_outputNum * this->_batch);
#endif
    }

    this->_layerDetail.append("yolov3. class num : " + std::to_string(classNum) + "\n");
}

void Yolov3Layer::forward(NetworkState &netState)
{
    auto st = TimeUtil::startRecord();

    Blas::cpuCopy(netState.inputNum, netState.input, 1, this->_output, 1);

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

    this->_forwardTime =   TimeUtil::getElapsedTime(st);
}

#ifdef USE_GPU
void Yolov3Layer::forwardGPU(NetworkState &netState)
{
    this->recordCudaStart();

    BlasGPU::gpuCopy(netState.inputNum, netState.input, 1, this->_gpuOutput, 1);

    int num     = this->_width*this->_height;
    int nxClass =  num * (1+this->_classNum);

    for (int b = 0; b < this->_batch; ++b)
    {
        for (int n = 0; n < 3; ++n)
        {

            int index = b*this->_outputNum + n*this->_width*this->_height*( 4 + 1 + this->_classNum);
            Yolov3LayerGPU::exSigmoidGpu(num, this->_gpuOutput + index, this->_width, this->_ratios, 1);

            index = index + this->_width*this->_height;
            Yolov3LayerGPU::exSigmoidGpu(num, this->_gpuOutput + index, this->_width, this->_ratios, 0);

            index = index + this->_width*this->_height;
            Yolov3LayerGPU::aExpTGpu(num, this->_gpuOutput + index, anchors[n%3*2]);

            index = index + this->_width*this->_height;
            Yolov3LayerGPU::aExpTGpu(num, this->_gpuOutput + index, anchors[n%3*2 + 1]);

            index = index + this->_width*this->_height;
            Yolov3LayerGPU::sigmoidGpu(nxClass, this->_gpuOutput + index);
        }

    }

    this->recordCudaStop();
}
#endif

void Yolov3Layer::sigmoid(float *val, const int &num)
{
#ifdef USE_OMP
#pragma omp parallel for num_threads(OMP_THREAD)
#endif
    for (int i = 0; i < num; ++i)
    {
        val[i] = 1.f/(1.f+expf(-val[i]));
    }
}

void Yolov3Layer::exSigmoid(float *val, const int &width, const int&height, const float &ratios, const bool &addGridW)
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
            val[i] = (1.f/(1.f+expf(-val[i])) + i/width)*ratios;        }
    }
}

void Yolov3Layer::aExpT(float *val, const int &num, const float &a)
{
#ifdef USE_OMP
#pragma omp parallel for num_threads(OMP_THREAD)
#endif
    for (int i = 0; i < num; ++i)
    {
        val[i] = a*expf(val[i]);
    }
}

int Yolov3Layer::getClassNum() const
{
    return _classNum;
}

int Yolov3Layer::getOrgHeight() const
{
    return _orgHeight;
}

int Yolov3Layer::getOrgWidth() const
{
    return _orgWidth;
}

float Yolov3Layer::getRatios() const
{
    return _ratios;
}
}
