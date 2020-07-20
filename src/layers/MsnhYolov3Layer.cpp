#include "Msnhnet/layers/MsnhYolov3Layer.h"

namespace Msnhnet
{
Yolov3Layer::Yolov3Layer(const int &batch, const int &width, const int &height, const int &num, const int &orgWidth, const int &orgHeight, const int &classNum, const std::vector<float> &anchors)
{
    this->type      =   LayerType::YOLOV3;
    this->layerName =   "Yolov3          ";
    this->num       =   num;
    this->batch     =   batch;
    this->height    =   height;
    this->width     =   width;
    this->channel   =   num;

    this->orgHeight =   orgHeight;
    this->orgWidth  =   orgWidth;

    this->classNum  =   classNum;

    this->outWidth  =   this->width;
    this->outHeight =   this->height;
    this->outChannel=   this->channel;

    this->outputNum =   this->height*this->width*this->num;
    this->inputNum  =   this->outputNum;

    this->anchors   =   anchors;

    this->ratios    =   1.f*orgHeight/outHeight;  

    if(3*(this->classNum + 4 + 1) != num)
    {
        throw Exception(1, "class num error!", __FILE__, __LINE__);
    }

    if(!BaseLayer::isPreviewMode)
    {
        this->output    =   new float[static_cast<size_t>(this->outputNum * this->batch)]();
    }

    this->layerDetail.append("yolov3. class num : " + std::to_string(classNum) + "\n");
}

void Yolov3Layer::forward(NetworkState &netState)
{
    auto st = std::chrono::system_clock::now();

    Blas::cpuCopy(netState.inputNum, netState.input, 1, this->output, 1);
#ifndef USE_GPU

    for (int b = 0; b < this->batch; ++b)
    {
        for (int n = 0; n < 3; ++n)
        {

            int index = b*this->outputNum + n*this->width*this->height*( 4 + 1 + this->classNum);
            exSigmoid(this->output + index, this->width, this->height, this->ratios, true);

            index = index + this->width*this->height;

            exSigmoid(this->output + index, this->width, this->height, this->ratios, false);

            index = index + this->width*this->height;

            aExpT(this->output + index, this->width*this->height, anchors[n%3*2]);

            index = index + this->width*this->height;
            aExpT(this->output + index, this->width*this->height, anchors[n%3*2 + 1]);

            index = index + this->width*this->height;
            sigmoid(this->output + index, this->width*this->height*(1+this->classNum));
        }

    }
    auto so = std::chrono::system_clock::now();
    this->forwardTime =   1.f * (std::chrono::duration_cast<std::chrono::microseconds>(so - st)).count()* std::chrono::microseconds::period::num / std::chrono::microseconds::period::den;
#endif
}

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
}
