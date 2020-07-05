#include "Msnhnet/layers/MsnhRouteLayer.h"

namespace Msnhnet
{
RouteLayer::RouteLayer(const int &batch, std::vector<int> &inputLayerIndexes,
                       std::vector<int> &inputLayerOutputs, const int &groups, const int &groupIndex)
{
    this->type              =   LayerType::ROUTE;
    this->layerName         =   "Route           ";

   this->batch             =   batch;
    this->groups            =   groups;
    this->groupIndex        =   groupIndex;
    int mOutputNum          =   0;

   this->layerDetail.append("route ");
    char msg[100];

   this->inputLayerIndexes =   inputLayerIndexes;
    this->inputLayerOutputs =   inputLayerOutputs;

   for (size_t i = 0; i < inputLayerIndexes.size(); ++i)
    {
#ifdef WIN32
        sprintf_s(msg, " %d", inputLayerIndexes[i]);
#else
        sprintf(msg, " %d", inputLayerIndexes[i]);
#endif
        this->layerDetail.append(msg);

       mOutputNum      =   mOutputNum + inputLayerOutputs[i];
    }

   this->layerDetail.append("\n");

   mOutputNum          =   mOutputNum / groups;
    this->outputNum     =   mOutputNum;
    this->inputNum      =   mOutputNum;

   if(!BaseLayer::isPreviewMode)
    {
        this->output        =   new float[static_cast<size_t>(this->outputNum*this->batch)]();
    }

}

void RouteLayer::forward(NetworkState &netState)        

{
    auto st = std::chrono::system_clock::now();
    int offset          =   0;
    for (size_t i = 0; i < inputLayerIndexes.size(); ++i)
    {
        int index       =   this->inputLayerIndexes[i];
        float *mInput   =   netState.net->layers[static_cast<size_t>(index)]->output;
        int inputLayerOutputs   =   this->inputLayerOutputs[i];
        int partInSize  =   inputLayerOutputs / this->groups;
        for (int j = 0; j < this->batch; ++j)
        {
            Blas::cpuCopy(partInSize, mInput + j*inputLayerOutputs + partInSize*this->groupIndex, 1,
                          this->output + offset + j*this->outputNum,1);
        }

       offset          = offset + partInSize;
    }
    auto so = std::chrono::system_clock::now();
    this->forwardTime =   1.f * (std::chrono::duration_cast<std::chrono::microseconds>(so - st)).count()* std::chrono::microseconds::period::num / std::chrono::microseconds::period::den;

}

void RouteLayer::resize(Network &net)
{
    BaseLayer first                 =   *net.layers[static_cast<size_t>(this->inputLayerIndexes[0])];
    this->outWidth                  =   first.outWidth;
    this->outHeight                 =   first.outHeight;
    this->outChannel                =   first.outChannel;
    this->outputNum                 =   first.outputNum;
    this->inputLayerOutputs[0]      =   first.outputNum;

   for (size_t i = 0; i < static_cast<size_t>(this->num); ++i)
    {
        size_t index                =   static_cast<size_t>(this->inputLayerIndexes[i]);
        BaseLayer next              =   *net.layers[index];
        this->outputNum             +=  next.outputNum;
        this->inputLayerOutputs[i]  = next.outputNum;

       if(next.outWidth == first.outWidth && next.outHeight == first.outHeight)
        {
            this->outChannel    +=  next.outChannel;
        }
        else
        {
            this->outHeight     =   0;
            this->outWidth      =   0;
            this->outChannel    =   0;
            throw Exception(1, "Different size of first layer and secon layer", __FILE__, __LINE__);
        }
    }

   this->outChannel    =   this->outChannel/this->groups;
    this->outputNum     =   this->outputNum / this->groups;
    this->inputNum      =   this->outputNum;
    if(this->output == nullptr)
    {
        throw Exception(1,"output can't be null", __FILE__, __LINE__);
    }

   this->output    = static_cast<float *>(realloc(this->output, static_cast<size_t>(this->outputNum *this->batch)*sizeof(float)));
}
}
