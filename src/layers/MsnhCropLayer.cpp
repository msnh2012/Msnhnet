#include "Msnhnet/layers/MsnhCropLayer.h"

namespace Msnhnet
{
CropLayer::CropLayer(const int &batch, const int &height, const int &width, const int &channel, const int &cropHeight, const int &cropWidth, const int &flip, const float &angle, const float &saturation, const float &exposure)
{
    (void)angle;
    this->_type          =   LayerType::CROP;
    this->_layerName     =  "Crop            ";

    this->_batch         =   batch;
    this->_height        =   height;
    this->_width         =   width;
    this->_channel       =   channel;
    this->_scale         =   static_cast<float>(cropHeight/height);
    this->_flip          =   flip;
    this->_saturation    =   saturation;
    this->_exposure      =   exposure;
    this->_outWidth      =   cropWidth;
    this->_outHeight     =   cropHeight;
    this->_outChannel    =   channel;

    this->_outputNum     =   this->_outWidth * this->_outHeight * this->_outChannel;
    this->_inputNum      =   width * height * channel;

    if(!BaseLayer::isPreviewMode)
    {
        this->_output        =   new float[static_cast<size_t>(this->_outputNum * this->_batch)]();
    }

    char msg[100];
#ifdef WIN32
    sprintf_s(msg, "Crop Layer: %d x %d -> %d x %d x %d image\n", height, width, cropHeight, cropWidth, channel);
#else
    sprintf(msg, "Crop Layer: %d x %d -> %d x %d x %d image\n", height, width, cropHeight, cropWidth, channel);
#endif
    this->_layerDetail   = msg;
}

void CropLayer::forward(NetworkState &netState)
{
    TimeUtil::startRecord();

    int flip        =   (this->_flip && rand() % 2);
    int dh          =   rand()%(this->_height - this->_outHeight + 1);
    int dw          =   rand()%(this->_width  - this->_outWidth  + 1);
    float scale     =   2;
    float trans     =  -1;
    int   col       =   0;
    int   row       =   0;
    int   index     =   0;
    int   cnt       =   0;

    if(this->_noAdjust)
    {
        scale       =   1;
        trans       =   0;
    }

    flip        =   0;
    dw          =   (this->_height - this->_outHeight)/2;
    dh          =   (this->_width  - this->_outWidth)/2;

    for (int b = 0; b < this->_batch; ++b)
    {
        for (int c = 0; c < this->_channel; ++c)
        {
            for (int i = 0; i < this->_outHeight; ++i)
            {
                for (int j = 0; j < this->_outWidth; ++j)
                {
                    if(flip)
                    {
                        col = this->_width - dw - j -1;
                    }
                    else
                    {
                        col = j + dw;
                    }

                    row     = i + dh;
                    index   = col + this->_width *(row + this->_height*(c + this->_channel*b));
                    this->_output[cnt++]   =   netState.input[index]*scale + trans;
                }
            }
        }
    }

    this->_forwardTime =   TimeUtil::getElapsedTime();

}

void CropLayer::resize(const int &width, const int &height)
{
    this->_width     =   width;
    this->_height    =   height;

    this->_outHeight =   _outHeight;
    this->_outWidth  =   _outWidth;

    this->_inputNum  =   this->_width * this->_height * this->_channel;
    this->_outputNum =   this->_outHeight * this->_outWidth * this->_outChannel;

    if(this->_output == nullptr)
    {
        throw Exception(1,"output can't be null", __FILE__, __LINE__, __FUNCTION__);
    }

    this->_output    = static_cast<float*>(realloc(this->_output, static_cast<size_t>(this->_batch*this->_outputNum)*sizeof(float)));
}

int CropLayer::getKSize() const
{
    return _kSize;
}

float CropLayer::getScale() const
{
    return _scale;
}

int CropLayer::getFlip() const
{
    return _flip;
}

float CropLayer::getSaturation() const
{
    return _saturation;
}

float CropLayer::getExposure() const
{
    return _exposure;
}

int CropLayer::getNoAdjust() const
{
    return _noAdjust;
}
}
