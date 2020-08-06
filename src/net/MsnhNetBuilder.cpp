#include "Msnhnet/net/MsnhNetBuilder.h"
namespace Msnhnet
{
NetBuilder::NetBuilder()
{
    parser          =   new Parser();
    net             =   new Network();
    netState        =   new NetworkState();
    netState->net   =   net;

    BaseLayer::initSimd();
}

NetBuilder::~NetBuilder()
{
    clearLayers();

    delete parser;
    parser      =   nullptr;

    delete netState;
    netState    =   nullptr;

    delete net;
    net         =   nullptr;

}

void NetBuilder::buildNetFromMsnhNet(const string &path)
{
    parser->readCfg(path);
    clearLayers();

    NetBuildParams      params;
    size_t      maxWorkSpace = 0;
    for (size_t i = 0; i < parser->params.size(); ++i)
    {
        BaseLayer   *layer;
        if(parser->params[i]->type == LayerType::CONFIG)
        {
            NetConfigParams* netCfgParams     =   reinterpret_cast<NetConfigParams*>(parser->params[i]);
            net->batch                  =   netCfgParams->batch;
            net->channels               =   netCfgParams->channels;
            net->width                  =   netCfgParams->width;
            net->height                 =   netCfgParams->height;

            if(netCfgParams->height == 0    || netCfgParams->height < 0||
                    netCfgParams->width == 0     || netCfgParams->width < 0 ||
                    netCfgParams->channels == 0  || netCfgParams->width < 0)
            {
                throw Exception(1,"net config params err, params = 0 or < 0", __FILE__, __LINE__, __FUNCTION__);
            }
            net->inputNum               =   net->batch * net->channels * net->width * net->height;

            params.height               =   net->height;
            params.batch                =   net->batch;
            params.width                =   net->width;
            params.channels             =   net->channels;
            params.inputNums            =   net->inputNum;
            continue;
        }

        if(parser->params[i]->type == LayerType::CONVOLUTIONAL)
        {
            if(params.height ==0 || params.width == 0 || params.channels == 0)
            {
                throw Exception(1, "Layer before convolutional layer must output image", __FILE__, __LINE__, __FUNCTION__);
            }

            ConvParams* convParams                  =   reinterpret_cast<ConvParams*>(parser->params[i]);
            layer                                   =   new ConvolutionalLayer(params.batch, 1, params.height, params.width, params.channels, convParams->filters,convParams->groups,
                                                                               convParams->kSizeX, convParams->kSizeY, convParams->strideX, convParams->strideY, convParams->dilationX,
                                                                               convParams->dilationY,convParams->paddingX, convParams->paddingY,
                                                                               convParams->activation, convParams->actParams, convParams->batchNorm, convParams->useBias,
                                                                               0,0,0,0,convParams->antialiasing, nullptr, 0,0);
        }
        else if(parser->params[i]->type == LayerType::ACTIVE)
        {
            ActivationParams *activationParams      =   reinterpret_cast<ActivationParams*>(parser->params[i]);
            layer                                   =   new ActivationLayer(params.batch, params.height, params.width, params.channels, params.inputNums, activationParams->activation);
        }
        else if(parser->params[i]->type == LayerType::DECONVOLUTIONAL)
        {
            if(params.height ==0 || params.width == 0 || params.channels == 0)
            {
                throw Exception(1, "Layer before deconvolutional layer must output image", __FILE__, __LINE__, __FUNCTION__);
            }

            DeConvParams* deconvParams              =   reinterpret_cast<DeConvParams*>(parser->params[i]);
            layer                                   =   new DeConvolutionalLayer(params.batch, params.height, params.width, params.channels, deconvParams->filters, deconvParams->kSizeX,
                                                                                 deconvParams->kSizeY, deconvParams->strideX, deconvParams->strideY, deconvParams->paddingX, deconvParams->paddingY,
                                                                                 deconvParams->groups, deconvParams->activation, deconvParams->actParams, deconvParams->useBias);
        }
        else if(parser->params[i]->type == LayerType::CONNECTED)
        {
            ConnectParams *connectParams            =   reinterpret_cast<ConnectParams*>(parser->params[i]);
            layer                                   =   new ConnectedLayer(params.batch, 1, params.inputNums, connectParams->output, connectParams->activation, connectParams->actParams,
                                                                           connectParams->batchNorm);
        }
        else if(parser->params[i]->type == LayerType::MAXPOOL)
        {
            MaxPoolParams *maxPoolParams            =   reinterpret_cast<MaxPoolParams*>(parser->params[i]);
            layer                                   =   new MaxPoolLayer(params.batch, params.height, params.width, params.channels, maxPoolParams->kSizeX, maxPoolParams->kSizeY,
                                                                         maxPoolParams->strideX, maxPoolParams->strideY, maxPoolParams->paddingX, maxPoolParams->paddingY,
                                                                         maxPoolParams->maxPoolDepth, maxPoolParams->outChannels, maxPoolParams->ceilMode, 0);
        }
        else if(parser->params[i]->type == LayerType::PADDING)
        {
            PaddingParams *paddingParams            =   reinterpret_cast<PaddingParams*>(parser->params[i]);
            layer                                   =   new PaddingLayer(params.batch, params.height, params.width, params.channels, paddingParams->top,
                                                                         paddingParams->down, paddingParams->left, paddingParams->right, paddingParams->paddingVal);
        }
        else if(parser->params[i]->type == LayerType::LOCAL_AVGPOOL)
        {
            LocalAvgPoolParams *localAvgPoolParams  =   reinterpret_cast<LocalAvgPoolParams*>(parser->params[i]);
            layer                                   =   new LocalAvgPoolLayer(params.batch, params.height, params.width, params.channels, localAvgPoolParams->kSizeX, localAvgPoolParams->kSizeY,
                                                                              localAvgPoolParams->strideX, localAvgPoolParams->strideY, localAvgPoolParams->paddingX, localAvgPoolParams->paddingY, localAvgPoolParams->ceilMode, 0);
        }
        else if(parser->params[i]->type == LayerType::GLOBAL_AVGPOOL)
        {
            layer                                   =   new GlobalAvgPoolLayer(params.batch, params.height, params.width, params.channels);
        }
        else if(parser->params[i]->type == LayerType::BATCHNORM)
        {
            BatchNormParams *batchNormParams        =   reinterpret_cast<BatchNormParams*>(parser->params[i]);
            layer                                   =   new BatchNormLayer(params.batch, params.width, params.height, params.channels, batchNormParams->activation, batchNormParams->actParams);
        }
        else if(parser->params[i]->type == LayerType::RES_BLOCK)
        {
            ResBlockParams *resBlockParams          =   reinterpret_cast<ResBlockParams*>(parser->params[i]);
            layer                                   =   new ResBlockLayer(params.batch, params, resBlockParams->baseParams, resBlockParams->activation, resBlockParams->actParams);
        }
        else if(parser->params[i]->type == LayerType::RES_2_BLOCK)
        {
            Res2BlockParams *res2BlockParams        =   reinterpret_cast<Res2BlockParams*>(parser->params[i]);
            layer                                   =   new Res2BlockLayer(params.batch, params, res2BlockParams->baseParams, res2BlockParams->branchParams, res2BlockParams->activation, res2BlockParams->actParams);
        }
        else if(parser->params[i]->type == LayerType::ADD_BLOCK)
        {
            AddBlockParams *addBlockParams          =   reinterpret_cast<AddBlockParams*>(parser->params[i]);
            layer                                   =   new AddBlockLayer(params.batch, params, addBlockParams->branchParams, addBlockParams->activation, addBlockParams->actParams);
        }
        else if(parser->params[i]->type == LayerType::CONCAT_BLOCK)
        {
            ConcatBlockParams *concatBlockParams    =   reinterpret_cast<ConcatBlockParams*>(parser->params[i]);
            layer                                   =   new ConcatBlockLayer(params.batch, params, concatBlockParams->branchParams, concatBlockParams->activation, concatBlockParams->actParams);
        }
        else if(parser->params[i]->type == LayerType::ROUTE)
        {
            RouteParams     *routeParams            =   reinterpret_cast<RouteParams*>(parser->params[i]);
            std::vector<int> layersOutputNum;

            if(routeParams->layerIndexes.size() < 1)
            {
                throw Exception(1, "route layer error, no route layers", __FILE__, __LINE__, __FUNCTION__);
            }
            int outChannel  =   0;

            size_t routeIndex = static_cast<size_t>(routeParams->layerIndexes[0]);

            if(routeIndex >= net->layers.size())
            {
                throw Exception(1, "route layer error, route layers index should < size of layers", __FILE__, __LINE__, __FUNCTION__);
            }

            int outHeight   =   net->layers[routeIndex]->getOutHeight();
            int outWidth    =   net->layers[routeIndex]->getOutWidth();

            for (size_t i = 0; i < routeParams->layerIndexes.size(); ++i)
            {
                size_t index   = static_cast<size_t>(routeParams->layerIndexes[i]);
                layersOutputNum.push_back(net->layers[index]->getOutputNum());

                if(outHeight != net->layers[index]->getOutHeight() || outWidth != net->layers[index]->getOutWidth())
                {
                    throw Exception(1, "[route] layers height or width not equal", __FILE__, __LINE__, __FUNCTION__);
                }

                if(routeParams->addModel == 1)
                {
                    outChannel = net->layers[index]->getOutChannel();
                }
                else
                {
                    outChannel +=   net->layers[index]->getOutChannel();
                }
            }
            layer                                   =   new RouteLayer(params.batch, routeParams->layerIndexes, layersOutputNum,
                                                                       routeParams->groups, routeParams->groupsId, routeParams->addModel);
            layer->setOutChannel(outChannel);
            layer->setOutWidth(outWidth);
            layer->setOutHeight(outHeight);
        }
        else if(parser->params[i]->type == LayerType::UPSAMPLE)
        {
            UpSampleParams *upSampleParams          =   reinterpret_cast<UpSampleParams*>(parser->params[i]);
            layer                                   =   new UpSampleLayer(params.batch, params.width, params.height, params.channels, upSampleParams->stride, upSampleParams->scale);
        }
        else if(parser->params[i]->type == LayerType::SOFTMAX)
        {
            SoftMaxParams  *softmaxParams           =   reinterpret_cast<SoftMaxParams*>(parser->params[i]);
            layer                                   =   new SoftMaxLayer(params.batch, params.inputNums, softmaxParams->groups, softmaxParams->temperature);
        }
        else if(parser->params[i]->type == LayerType::YOLOV3)
        {
            Yolov3Params *yolov3Params              =   reinterpret_cast<Yolov3Params*>(parser->params[i]);
            layer                                   =   new Yolov3Layer(params.batch, params.width, params.height, params.channels, yolov3Params->orgWidth, yolov3Params->orgHeight,
                                                                        yolov3Params->classNum, yolov3Params->anchors);
        }
        else if(parser->params[i]->type == LayerType::YOLOV3_OUT)
        {
            Yolov3OutParams     *yolov3OutParams    =   reinterpret_cast<Yolov3OutParams*>(parser->params[i]);

            std::vector<Yolov3Info> yolov3LayersInfo;

            if(yolov3OutParams->layerIndexes.size() < 1)
            {
                throw Exception(1, "yolov3out layer error, no yolov3 layers", __FILE__, __LINE__, __FUNCTION__);
            }

            for (size_t i = 0; i < yolov3OutParams->layerIndexes.size(); ++i)
            {
                size_t index   =   static_cast<size_t>(yolov3OutParams->layerIndexes[i]);

                if(net->layers[index]->type() != LayerType::YOLOV3)
                {
                    throw Exception(1, "yolov3out layer error, not a yolov3 layer", __FILE__, __LINE__, __FUNCTION__);
                }

                yolov3LayersInfo.push_back(Yolov3Info(net->layers[index]->getOutHeight(),
                                                      net->layers[index]->getOutWidth(),
                                                      net->layers[index]->getOutChannel()
                                                      ));
            }

            layer                                   =   new Yolov3OutLayer(params.batch, yolov3OutParams->orgWidth, yolov3OutParams->orgHeight, yolov3OutParams->layerIndexes,
                                                                           yolov3LayersInfo,yolov3OutParams->confThresh, yolov3OutParams->nmsThresh, yolov3OutParams->useSoftNms, yolov3OutParams->yoloType);
        }

        params.height       =   layer->getOutHeight();
        params.width        =   layer->getOutWidth();
        params.channels     =   layer->getOutChannel();
        params.inputNums    =   layer->getOutputNum();

        if(layer->getWorkSpaceSize() > maxWorkSpace)
        {
            maxWorkSpace = layer->getWorkSpaceSize();
        }
        net->layers.push_back(layer);
    }

    netState->workspace     =   new float[maxWorkSpace]();
#ifdef USE_GPU
    netState->gpuWorkspace  =   Cuda::makeCudaArray(netState->workspace,maxWorkSpace);
#endif
}

void NetBuilder::loadWeightsFromMsnhBin(const string &path)
{
    if(BaseLayer::isPreviewMode)
    {
        throw Exception(1, "Can not load weights in preview mode !",__FILE__, __LINE__, __FUNCTION__);
    }

    parser->readMsnhBin(path);
    size_t ptr = 0;
    std::vector<float>::const_iterator first = parser->msnhF32Weights.begin();

    for (size_t i = 0; i < net->layers.size(); ++i)
    {
        if(net->layers[i]->type() == LayerType::CONVOLUTIONAL || net->layers[i]->type() == LayerType::CONNECTED || net->layers[i]->type() == LayerType::BATCHNORM ||
                net->layers[i]->type() == LayerType::RES_BLOCK   || net->layers[i]->type() == LayerType::RES_2_BLOCK || net->layers[i]->type() == LayerType::ADD_BLOCK ||
                net->layers[i]->type() == LayerType::CONCAT_BLOCK  || net->layers[i]->type() == LayerType::DECONVOLUTIONAL)
        {
            size_t nums = net->layers[i]->getNumWeights();

            if((ptr + nums) > (parser->msnhF32Weights.size()))
            {
                throw Exception(1,"Load weights err, need > given. Needed :" + std::to_string(ptr + nums) + "given :" +
                                std::to_string(parser->msnhF32Weights.size()),__FILE__,__LINE__, __FUNCTION__);
            }

            std::vector<float> weights(first +  static_cast<long long>(ptr), first + static_cast<long long>(ptr + nums));

            net->layers[i]->loadAllWeigths(weights);

            ptr         =   ptr + nums;
        }
    }

    if(ptr != parser->msnhF32Weights.size())
    {
        throw Exception(1,"Load weights err, need != given. Needed :" + std::to_string(ptr) + "given :" +
                        std::to_string(parser->msnhF32Weights.size()),__FILE__,__LINE__, __FUNCTION__);
    }

}

void NetBuilder::setPreviewMode(const bool &mode)
{
    BaseLayer::setPreviewMode(mode);
}

std::vector<float> NetBuilder::runClassify(std::vector<float> img)
{
    if(BaseLayer::isPreviewMode)
    {
        throw Exception(1, "Can not infer in preview mode !",__FILE__, __LINE__, __FUNCTION__);
    }
#ifdef USE_NNPACK
    nnp_initialize();
#endif
    netState->input     =   img.data();
    netState->inputNum  =   static_cast<int>(img.size());
    if(net->layers[0]->getInputNum() != netState->inputNum)
    {
        throw Exception(1,"input image size err. Needed :" + std::to_string(net->layers[0]->getInputNum()) + "given :" +
                std::to_string(img.size()),__FILE__,__LINE__, __FUNCTION__);
    }

    for (size_t i = 0; i < net->layers.size(); ++i)
    {
        net->layers[i]->forward(*netState);

        netState->input     =   net->layers[i]->getOutput();
        netState->inputNum  =   net->layers[i]->getOutputNum();

    }

#ifdef USE_NNPACK
    nnp_deinitialize();
#endif
    std::vector<float> pred(netState->input, netState->input + netState->inputNum);

    return pred;
}

std::vector<std::vector<Yolov3Box>> NetBuilder::runYolov3(std::vector<float> img)
{
    if(BaseLayer::isPreviewMode)
    {
        throw Exception(1, "Can not infer in preview mode !",__FILE__, __LINE__, __FUNCTION__);
    }
#ifdef USE_NNPACK
    nnp_initialize();
#endif
    netState->input     =   img.data();
    netState->inputNum  =   static_cast<int>(img.size());
    if(net->layers[0]->getInputNum() != netState->inputNum)
    {
        throw Exception(1,"input image size err. Needed :" + std::to_string(net->layers[0]->getInputNum()) + "given :" +
                std::to_string(img.size()),__FILE__,__LINE__, __FUNCTION__);
    }

    for (size_t i = 0; i < net->layers.size(); ++i)
    {

        if(net->layers[i]->type() != LayerType::ROUTE && net->layers[i]->type() != LayerType::YOLOV3_OUT) 

        {
            if(netState->inputNum != net->layers[i]->getInputNum())
            {
                throw Exception(1, "layer " + to_string(i) + " inputNum needed : " + std::to_string(net->layers[i]->getInputNum()) +
                                ", given : " + std::to_string(netState->inputNum),__FILE__,__LINE__, __FUNCTION__);
            }
        }

        net->layers[i]->forward(*netState);

        netState->input     =   net->layers[i]->getOutput();
        netState->inputNum  =   net->layers[i]->getOutputNum();

    }

#ifdef USE_NNPACK
    nnp_deinitialize();
#endif

    if((net->layers[net->layers.size()-1])->type() == LayerType::YOLOV3_OUT)
    {
        return (reinterpret_cast<Yolov3OutLayer*>((net->layers[net->layers.size()-1])))->finalOut;
    }
    else
    {
        throw Exception(1,"not a yolov3 net", __FILE__, __LINE__, __FUNCTION__);
    }
}

#ifdef USE_GPU
std::vector<float> NetBuilder::runClassifyGPU(std::vector<float> img)
{
    if(BaseLayer::isPreviewMode)
    {
        throw Exception(1, "Can not infer in preview mode !",__FILE__, __LINE__, __FUNCTION__);
    }

    netState->inputNum  =   static_cast<int>(img.size());

    if(net->layers[0]->getInputNum() != netState->inputNum)
    {
        throw Exception(1,"input image size err. Needed :" + std::to_string(net->layers[0]->getInputNum()) + "given :" +
                std::to_string(img.size()),__FILE__,__LINE__, __FUNCTION__);
    }

    netState->input     =   Cuda::makeCudaArray(img.data(), img.size());

    for (size_t i = 0; i < net->layers.size(); ++i)
    {
        net->layers[i]->forwardGPU(*netState);
        netState->input     =   net->layers[i]->getGpuOutput();
        netState->inputNum  =   net->layers[i]->getOutputNum();
    }
    float* out = new float[netState->inputNum]();
    Cuda::pullCudaArray(netState->input, out,netState->inputNum);
    std::vector<float> pred(out, out + netState->inputNum);
    delete[] out;
    return pred;
}

std::vector<std::vector<Yolov3Box>> NetBuilder::runYolov3GPU(std::vector<float> img)
{
    if(BaseLayer::isPreviewMode)
    {
        throw Exception(1, "Can not infer in preview mode !",__FILE__, __LINE__, __FUNCTION__);
    }

    netState->inputNum  =   static_cast<int>(img.size());

    if(net->layers[0]->getInputNum() != netState->inputNum)
    {
        throw Exception(1,"input image size err. Needed :" + std::to_string(net->layers[0]->getInputNum()) + "given :" +
                std::to_string(img.size()),__FILE__,__LINE__, __FUNCTION__);
    }

    netState->input     =   Cuda::makeCudaArray(img.data(), img.size());

    for (size_t i = 0; i < net->layers.size(); ++i)
    {

        if(net->layers[i]->type() != LayerType::ROUTE && net->layers[i]->type() != LayerType::YOLOV3_OUT) 

        {
            if(netState->inputNum != net->layers[i]->getInputNum())
            {
                throw Exception(1, "layer " + to_string(i) + " inputNum needed : " + std::to_string(net->layers[i]->getInputNum()) +
                                ", given : " + std::to_string(netState->inputNum),__FILE__,__LINE__, __FUNCTION__);
            }
        }

        net->layers[i]->forwardGPU(*netState);

        netState->input     =   net->layers[i]->getGpuOutput();
        netState->inputNum  =   net->layers[i]->getOutputNum();

    }

    if((net->layers[net->layers.size()-1])->type() == LayerType::YOLOV3_OUT)
    {
        return (reinterpret_cast<Yolov3OutLayer*>((net->layers[net->layers.size()-1])))->finalOut;
    }
    else
    {
        throw Exception(1,"not a yolov3 net", __FILE__, __LINE__, __FUNCTION__);
    }
}

#endif

Point2I NetBuilder::getInputSize()
{
    if(parser->params.empty())
    {
        throw Exception(1,"net param is empty", __FILE__, __LINE__, __FUNCTION__);
    }

    if(parser->params[0]->type == LayerType::CONFIG)
    {
        NetConfigParams* params = reinterpret_cast<NetConfigParams*>(parser->params[0]);
        return Point2I(params->width,params->height);
    }
    else
    {
        throw Exception(1,"net param error", __FILE__, __LINE__, __FUNCTION__);
    }
}

void NetBuilder::clearLayers()
{
    for (size_t i = 0; i < net->layers.size(); ++i)
    {
        if(net->layers[i]!=nullptr)
        {
            if(net->layers[i]->type() == LayerType::CONVOLUTIONAL)
            {
                delete reinterpret_cast<ConvolutionalLayer*>(net->layers[i]);
            }
            else if(net->layers[i]->type() == LayerType::MAXPOOL)
            {
                delete reinterpret_cast<MaxPoolLayer*>(net->layers[i]);
            }
            else if(net->layers[i]->type() == LayerType::ACTIVE)
            {
                delete reinterpret_cast<ActivationLayer*>(net->layers[i]);
            }
            else if(net->layers[i]->type() == LayerType::SOFTMAX)
            {
                delete reinterpret_cast<SoftMaxLayer*>(net->layers[i]);
            }
            else if(net->layers[i]->type() == LayerType::CONNECTED)
            {
                delete reinterpret_cast<ConnectedLayer*>(net->layers[i]);
            }
            else if(net->layers[i]->type() == LayerType::BATCHNORM)
            {
                delete reinterpret_cast<BatchNormLayer*>(net->layers[i]);
            }
            else if(net->layers[i]->type() == LayerType::LOCAL_AVGPOOL)
            {
                delete reinterpret_cast<LocalAvgPoolLayer*>(net->layers[i]);
            }
            else if(net->layers[i]->type() == LayerType::GLOBAL_AVGPOOL)
            {
                delete reinterpret_cast<GlobalAvgPoolLayer*>(net->layers[i]);
            }
            else if(net->layers[i]->type() == LayerType::RES_BLOCK)
            {
                delete reinterpret_cast<ResBlockLayer*>(net->layers[i]);
            }
            else if(net->layers[i]->type() == LayerType::RES_2_BLOCK)
            {
                delete reinterpret_cast<Res2BlockLayer*>(net->layers[i]);
            }
            else if(net->layers[i]->type() == LayerType::ADD_BLOCK)
            {
                delete reinterpret_cast<AddBlockLayer*>(net->layers[i]);
            }
            else if(net->layers[i]->type() == LayerType::CONCAT_BLOCK)
            {
                delete reinterpret_cast<ConcatBlockLayer*>(net->layers[i]);
            }
            else if(net->layers[i]->type() == LayerType::ROUTE)
            {
                delete reinterpret_cast<RouteLayer*>(net->layers[i]);
            }
            else if(net->layers[i]->type() == LayerType::UPSAMPLE)
            {
                delete reinterpret_cast<UpSampleLayer*>(net->layers[i]);
            }
            else if(net->layers[i]->type() == LayerType::YOLOV3)
            {
                delete reinterpret_cast<Yolov3Layer*>(net->layers[i]);
            }
            else if(net->layers[i]->type() == LayerType::YOLOV3_OUT)
            {
                delete reinterpret_cast<Yolov3OutLayer*>(net->layers[i]);
            }
            else if(net->layers[i]->type() == LayerType::PADDING)
            {
                delete reinterpret_cast<PaddingLayer*>(net->layers[i]);
            }

            net->layers[i] = nullptr;
        }

        if(i == (net->layers.size()-1))
        {
            net->layers.clear();
        }
    }
}

float NetBuilder::getInferenceTime()
{
    float inferTime     =   0.f;
    for (size_t i = 0; i < this->net->layers.size(); ++i)
    {
        inferTime       +=  this->net->layers[i]->getForwardTime();
    }

    return inferTime;
}

string NetBuilder::getLayerDetail()
{
    std::string detail;
    for(size_t i=0;i<this->net->layers.size();++i)
    {
        detail = detail + "────────────────────────────────  " + ((i<10)?("00"+std::to_string(i)):((i<100)?("0"+std::to_string(i)):std::to_string(i)))
                + " ─────────────────────────────────\n";
        detail = detail + this->net->layers[i]->getLayerDetail()  + "weights : "+
                std::to_string(this->net->layers[i]->getNumWeights()) + "\n\n";

    }
    return detail;
}

string NetBuilder::getTimeDetail()
{
    float totalWaste = getInferenceTime();
    std::string detail;
    detail     = detail + "LAYER            INDEX       TIME         LAYER_t/TOTAL_t\n"
            + "=========================================================\n";

    for(size_t i=0;i<this->net->layers.size();++i)
    {
        detail = detail + this->net->layers[i]->getLayerName() + " : ";
        detail = detail + ((i<10)?("00"+std::to_string(i)):((i<100)?("0"+std::to_string(i)):std::to_string(i))) + "     ";
        detail = detail + Msnhnet::ExString::left(std::to_string(this->net->layers[i]->getForwardTime()),6) +" ms         ";
        detail = detail + Msnhnet::ExString::left(std::to_string(((int)(this->net->layers[i]->getForwardTime() / totalWaste *1000))/10.f),4) + "%\n";
    }
    detail     = detail + "=========================================================\n";
    detail     = detail + "Msnhnet inference time : " + std::to_string(totalWaste) + " ms";
    return detail;
}

}
