#include "Msnhnet/net/MsnhNetBuilder.h"
namespace Msnhnet
{
NetBuilder::NetBuilder()
{
    _parser          =   new Parser();
    _net             =   new Network();
    _netState        =   new NetworkState();
    _netState->net   =   _net;

    BaseLayer::initSimd();
}

NetBuilder::~NetBuilder()
{
    clearLayers();

    delete _parser;
    _parser      =   nullptr;

    delete _netState;
    _netState    =   nullptr;

    delete _net;
    _net         =   nullptr;

}

void NetBuilder::buildNetFromMsnhNet(const string &path)
{
    _parser->readCfg(path);
    clearLayers();

    if(BaseLayer::useFp16)
    {
        std::cout<<"CUDNN USE FP16"<<std::endl<<std::endl;
    }
    NetBuildParams      params;
    size_t      maxWorkSpace    = 0;
    size_t      maxInputSpace   = 0;
    for (size_t i = 0; i < _parser->params.size(); ++i)
    {
        BaseLayer   *layer;
        if(_parser->params[i]->type == LayerType::CONFIG)
        {
            NetConfigParams* netCfgParams     =   reinterpret_cast<NetConfigParams*>(_parser->params[i]);
            _net->batch                  =   netCfgParams->batch;
            _net->channels               =   netCfgParams->channels;
            _net->width                  =   netCfgParams->width;
            _net->height                 =   netCfgParams->height;

            if(netCfgParams->height == 0    || netCfgParams->height < 0||
                    netCfgParams->width == 0     || netCfgParams->width < 0 ||
                    netCfgParams->channels == 0  || netCfgParams->width < 0)
            {
                throw Exception(1,"net config params err, params = 0 or < 0", __FILE__, __LINE__, __FUNCTION__);
            }
            _net->inputNum               =   _net->batch * _net->channels * _net->width * _net->height;

            params.height               =   _net->height;
            params.batch                =   _net->batch;
            params.width                =   _net->width;
            params.channels             =   _net->channels;
            params.inputNums            =   _net->inputNum;
            continue;
        }

        if(_parser->params[i]->type == LayerType::CONVOLUTIONAL)
        {
            if(params.height ==0 || params.width == 0 || params.channels == 0)
            {
                throw Exception(1, "Layer before convolutional layer must output image", __FILE__, __LINE__, __FUNCTION__);
            }

            ConvParams* convParams                  =   reinterpret_cast<ConvParams*>(_parser->params[i]);
            layer                                   =   new ConvolutionalLayer(params.batch, 1, params.height, params.width, params.channels, convParams->filters,convParams->groups,
                                                                               convParams->kSizeX, convParams->kSizeY, convParams->strideX, convParams->strideY, convParams->dilationX,
                                                                               convParams->dilationY,convParams->paddingX, convParams->paddingY,
                                                                               convParams->activation, convParams->actParams, convParams->batchNorm, convParams->useBias,
                                                                               0,0,0,0,convParams->antialiasing, nullptr, 0,0);
        }
        else if(_parser->params[i]->type == LayerType::ACTIVE)
        {
            ActivationParams *activationParams      =   reinterpret_cast<ActivationParams*>(_parser->params[i]);
            layer                                   =   new ActivationLayer(params.batch, params.height, params.width, params.channels, params.inputNums, activationParams->activation);
        }
        else if(_parser->params[i]->type == LayerType::DECONVOLUTIONAL)
        {
            if(params.height ==0 || params.width == 0 || params.channels == 0)
            {
                throw Exception(1, "Layer before deconvolutional layer must output image", __FILE__, __LINE__, __FUNCTION__);
            }

            DeConvParams* deconvParams              =   reinterpret_cast<DeConvParams*>(_parser->params[i]);
            layer                                   =   new DeConvolutionalLayer(params.batch, params.height, params.width, params.channels, deconvParams->filters, deconvParams->kSizeX,
                                                                                 deconvParams->kSizeY, deconvParams->strideX, deconvParams->strideY, deconvParams->paddingX, deconvParams->paddingY,
                                                                                 deconvParams->groups, deconvParams->activation, deconvParams->actParams, deconvParams->useBias);
        }
        else if(_parser->params[i]->type == LayerType::CONNECTED)
        {
            ConnectParams *connectParams            =   reinterpret_cast<ConnectParams*>(_parser->params[i]);
            layer                                   =   new ConnectedLayer(params.batch, 1, params.inputNums, connectParams->output, connectParams->activation, connectParams->actParams,
                                                                           connectParams->batchNorm, connectParams->useBias);
        }
        else if(_parser->params[i]->type == LayerType::MAXPOOL)
        {
            MaxPoolParams *maxPoolParams            =   reinterpret_cast<MaxPoolParams*>(_parser->params[i]);
            layer                                   =   new MaxPoolLayer(params.batch, params.height, params.width, params.channels, maxPoolParams->kSizeX, maxPoolParams->kSizeY,
                                                                         maxPoolParams->strideX, maxPoolParams->strideY, maxPoolParams->paddingX, maxPoolParams->paddingY,
                                                                         maxPoolParams->maxPoolDepth, maxPoolParams->outChannels, maxPoolParams->ceilMode, 0);
        }
        else if(_parser->params[i]->type == LayerType::PADDING)
        {
            PaddingParams *paddingParams            =   reinterpret_cast<PaddingParams*>(_parser->params[i]);
            layer                                   =   new PaddingLayer(params.batch, params.height, params.width, params.channels, paddingParams->top,
                                                                         paddingParams->down, paddingParams->left, paddingParams->right, paddingParams->paddingVal);
        }
        else if(_parser->params[i]->type == LayerType::LOCAL_AVGPOOL)
        {
            LocalAvgPoolParams *localAvgPoolParams  =   reinterpret_cast<LocalAvgPoolParams*>(_parser->params[i]);
            layer                                   =   new LocalAvgPoolLayer(params.batch, params.height, params.width, params.channels, localAvgPoolParams->kSizeX, localAvgPoolParams->kSizeY,
                                                                              localAvgPoolParams->strideX, localAvgPoolParams->strideY, localAvgPoolParams->paddingX, localAvgPoolParams->paddingY, localAvgPoolParams->ceilMode, 0);
        }
        else if(_parser->params[i]->type == LayerType::GLOBAL_AVGPOOL)
        {
            layer                                   =   new GlobalAvgPoolLayer(params.batch, params.height, params.width, params.channels);
        }
        else if(_parser->params[i]->type == LayerType::BATCHNORM)
        {
            BatchNormParams *batchNormParams        =   reinterpret_cast<BatchNormParams*>(_parser->params[i]);
            layer                                   =   new BatchNormLayer(params.batch, params.width, params.height, params.channels, batchNormParams->activation, batchNormParams->actParams);
        }
        else if(_parser->params[i]->type == LayerType::RES_BLOCK)
        {
            ResBlockParams *resBlockParams          =   reinterpret_cast<ResBlockParams*>(_parser->params[i]);
            layer                                   =   new ResBlockLayer(params.batch, params, resBlockParams->baseParams, resBlockParams->activation, resBlockParams->actParams);
        }
        else if(_parser->params[i]->type == LayerType::RES_2_BLOCK)
        {
            Res2BlockParams *res2BlockParams        =   reinterpret_cast<Res2BlockParams*>(_parser->params[i]);
            layer                                   =   new Res2BlockLayer(params.batch, params, res2BlockParams->baseParams, res2BlockParams->branchParams, res2BlockParams->activation, res2BlockParams->actParams);
        }
        else if(_parser->params[i]->type == LayerType::ADD_BLOCK)
        {
            AddBlockParams *addBlockParams          =   reinterpret_cast<AddBlockParams*>(_parser->params[i]);
            layer                                   =   new AddBlockLayer(params.batch, params, addBlockParams->branchParams, addBlockParams->activation, addBlockParams->actParams);
        }
        else if(_parser->params[i]->type == LayerType::CONCAT_BLOCK)
        {
            ConcatBlockParams *concatBlockParams    =   reinterpret_cast<ConcatBlockParams*>(_parser->params[i]);
            layer                                   =   new ConcatBlockLayer(params.batch, params, concatBlockParams->branchParams, concatBlockParams->activation, concatBlockParams->actParams);
        }
        else if(_parser->params[i]->type == LayerType::ROUTE)
        {
            RouteParams     *routeParams            =   reinterpret_cast<RouteParams*>(_parser->params[i]);
            std::vector<int> layersOutputNum;

            if(routeParams->layerIndexes.size() < 1)
            {
                throw Exception(1, "route layer error, no route layers", __FILE__, __LINE__, __FUNCTION__);
            }

            int outChannel  =   0;
            int outHeight   =   0;
            int outWidth    =   0;

            if(routeParams->layerIndexes.size() > 1)
            {
                for (size_t k = 0; k < routeParams->layerIndexes.size() - 1; ++k)
                {

                    size_t routeIndex       = static_cast<size_t>(routeParams->layerIndexes[k]);
                    size_t routeIndexNext   = static_cast<size_t>(routeParams->layerIndexes[k+1]);

                    if(routeIndex >= _net->layers.size() || routeIndexNext >= _net->layers.size())
                    {
                        throw Exception(1, "route layer error, route layers index should < size of layers", __FILE__, __LINE__, __FUNCTION__);
                    }

                    int tmpHeight       =   _net->layers[routeIndex]->getOutHeight();
                    int tmpWidth        =   _net->layers[routeIndex]->getOutWidth();
                    int tmpCh           =   _net->layers[routeIndex]->getOutChannel();

                    int tmpHeightNext   =   _net->layers[routeIndexNext]->getOutHeight();
                    int tmpWidthNext    =   _net->layers[routeIndexNext]->getOutWidth();
                    int tmpChNext       =   _net->layers[routeIndexNext]->getOutChannel();

                    if(routeParams->addModel == 1)
                    {
                        if(tmpHeight != tmpHeightNext || tmpWidth != tmpWidthNext || tmpCh != tmpChNext)
                        {
                            std::string indexes = "";
                            for (size_t k = 0; k < routeParams->layerIndexes.size(); ++k)
                            {
                                indexes += std::to_string(routeParams->layerIndexes[k]) + ", ";
                            }

                            throw Exception(1, "[route] layers height or width not equal. layer: "+ std::to_string(i-1) +
                                            + "  aux: " + indexes +
                                            "whc: "+
                                            std::to_string(tmpWidth) +
                                            ":" + std::to_string(tmpWidthNext) + " , " +
                                            std::to_string(tmpHeight) +
                                            ":" + std::to_string(tmpHeightNext) + " , " +
                                            std::to_string(tmpCh) +
                                            ":" + std::to_string(tmpChNext) + " "
                                            , __FILE__, __LINE__, __FUNCTION__);
                        }

                        outChannel = tmpCh;

                    }
                    else
                    {
                        if(tmpHeight != tmpHeightNext || tmpWidth != tmpWidthNext)
                        {
                            std::string indexes = "";
                            for (size_t k = 0; k < routeParams->layerIndexes.size(); ++k)
                            {
                                indexes += std::to_string(routeParams->layerIndexes[k]) + ", ";
                            }

                            throw Exception(1, "[route] layers height or width not equal. layer: "+ std::to_string(i-1) +
                                            + "  aux: " + indexes +
                                            "wh: "+
                                            std::to_string(tmpWidth)+
                                            ":" + std::to_string(tmpWidthNext) + " , " +
                                            std::to_string(tmpHeight)+
                                            ":" + std::to_string(tmpHeightNext) + " "
                                            , __FILE__, __LINE__, __FUNCTION__);
                        }

                        outChannel += tmpCh;

                        if(k == routeParams->layerIndexes.size() - 2) 

                        {
                            outChannel += tmpChNext;
                        }
                    }

                    outWidth    = tmpWidth;
                    outHeight   = tmpHeight;
                    layersOutputNum.push_back(_net->layers[routeIndex]->getOutputNum());
                    if(k == routeParams->layerIndexes.size() - 2) 

                    {
                        layersOutputNum.push_back(_net->layers[routeIndexNext]->getOutputNum());
                    }
                }
            }
            else
            {
                if(routeParams->addModel == 1)
                {
                    throw Exception(1, "route layer error, add model must have at least 2 layers ", __FILE__, __LINE__, __FUNCTION__);
                }
                else
                {
                    size_t routeIndex       = static_cast<size_t>(routeParams->layerIndexes[0]);
                    layersOutputNum.push_back(_net->layers[routeIndex]->getOutputNum());
                    outHeight       =   _net->layers[routeIndex]->getOutHeight();
                    outWidth        =   _net->layers[routeIndex]->getOutWidth();
                    outChannel      =   _net->layers[routeIndex]->getOutChannel();
                }
            }

            layer                                   =   new RouteLayer(params.batch, routeParams->layerIndexes, layersOutputNum,
                                                                       routeParams->groups, routeParams->groupsId, routeParams->addModel, routeParams->activation, routeParams->actParams);
            layer->setOutChannel(outChannel);
            layer->setOutWidth(outWidth);
            layer->setOutHeight(outHeight);
        }
        else if(_parser->params[i]->type == LayerType::VARIABLE_OP)
        {
            VariableOpParams     *variableOpParams            =   reinterpret_cast<VariableOpParams*>(_parser->params[i]);

            if(variableOpParams->varOpType == VariableOpParams::VarOpType::VAR_OP_ADD ||
                    variableOpParams->varOpType == VariableOpParams::VarOpType::VAR_OP_SUB ||
                    variableOpParams->varOpType == VariableOpParams::VarOpType::VAR_OP_SUB_INV ||
                    variableOpParams->varOpType == VariableOpParams::VarOpType::VAR_OP_MUL ||
                    variableOpParams->varOpType == VariableOpParams::VarOpType::VAR_OP_DIV ||
                    variableOpParams->varOpType == VariableOpParams::VarOpType::VAR_OP_DIV_INV
                    )
            {
                if(variableOpParams->layerIndexes.size() !=1 && variableOpParams->layerIndexes.size() != 2)
                {
                    throw Exception(1, "VarOp layer error, <add sub mul div> mode must have 1/2 extra layer(s)  ", __FILE__, __LINE__, __FUNCTION__);
                }

                if(variableOpParams->layerIndexes.size() == 2)
                {
                    for (int k = 0; k < 2; ++k)
                    {
                        if(variableOpParams->layerIndexes[k] == i-2)
                        {
                            throw Exception(1, "VarOp layer error, <add sub mul div> front layer should not contains in layers option", __FILE__, __LINE__, __FUNCTION__);
                        }
                    }
                }

                if(variableOpParams->layerIndexes.size() == 1)
                {
                    size_t routeIndex       = static_cast<size_t>(variableOpParams->layerIndexes[0]);

                    int tmpHeight       =   _net->layers[routeIndex]->getOutHeight();
                    int tmpWidth        =   _net->layers[routeIndex]->getOutWidth();
                    int tmpCh           =   _net->layers[routeIndex]->getOutChannel();

                    if(tmpHeight != params.height || tmpWidth != params.width || tmpCh != params.channels)
                    {
                        std::string indexes = std::to_string(variableOpParams->layerIndexes[0]);

                        throw Exception(1, "[VarOp] layers height or width not equal. layer: "+ std::to_string(i-1) +
                                        + "  aux: " + indexes +
                                        "whc: "+
                                        std::to_string(tmpWidth) +
                                        ":" + std::to_string(params.width) + " , " +
                                        std::to_string(tmpHeight) +
                                        ":" + std::to_string(params.height) + " , " +
                                        std::to_string(tmpCh) +
                                        ":" + std::to_string(params.channels) + " "
                                        , __FILE__, __LINE__, __FUNCTION__);
                    }
                }
                else
                {
                    size_t routeIndex       = static_cast<size_t>(variableOpParams->layerIndexes[0]);
                    size_t routeIndex1      = static_cast<size_t>(variableOpParams->layerIndexes[1]);

                    int tmpHeight       =   _net->layers[routeIndex]->getOutHeight();
                    int tmpWidth        =   _net->layers[routeIndex]->getOutWidth();
                    int tmpCh           =   _net->layers[routeIndex]->getOutChannel();

                    int tmpHeight1      =   _net->layers[routeIndex1]->getOutHeight();
                    int tmpWidth1       =   _net->layers[routeIndex1]->getOutWidth();
                    int tmpCh1          =   _net->layers[routeIndex1]->getOutChannel();

                    if(tmpHeight != tmpHeight1 || tmpWidth != tmpWidth1|| tmpCh != tmpCh1)
                    {
                        std::string indexes = std::to_string(variableOpParams->layerIndexes[0]) + " : " + std::to_string(variableOpParams->layerIndexes[1]);

                        throw Exception(1, "[VarOp] layers height or width not equal. layer: "+ std::to_string(i-1) +
                                        + "  aux: " + indexes +
                                        "whc: "+
                                        std::to_string(tmpWidth) +
                                        ":" + std::to_string(tmpWidth1) + " , " +
                                        std::to_string(tmpHeight) +
                                        ":" + std::to_string(tmpHeight1) + " , " +
                                        std::to_string(tmpCh) +
                                        ":" + std::to_string(tmpCh1) + " "
                                        , __FILE__, __LINE__, __FUNCTION__);
                    }
                }
            }
            else if(variableOpParams->varOpType == VariableOpParams::VarOpType::VAR_OP_ADD_CONST ||
                    variableOpParams->varOpType == VariableOpParams::VarOpType::VAR_OP_SUB_CONST ||
                    variableOpParams->varOpType == VariableOpParams::VarOpType::VAR_OP_SUB_CONST_INV ||
                    variableOpParams->varOpType == VariableOpParams::VarOpType::VAR_OP_MUL_CONST ||
                    variableOpParams->varOpType == VariableOpParams::VarOpType::VAR_OP_DIV_CONST ||
                    variableOpParams->varOpType == VariableOpParams::VarOpType::VAR_OP_DIV_CONST_INV||
                    variableOpParams->varOpType == VariableOpParams::VarOpType::VAR_OP_ABS||
                    variableOpParams->varOpType == VariableOpParams::VarOpType::VAR_OP_ACOS||
                    variableOpParams->varOpType == VariableOpParams::VarOpType::VAR_OP_ASIN||
                    variableOpParams->varOpType == VariableOpParams::VarOpType::VAR_OP_ATAN||
                    variableOpParams->varOpType == VariableOpParams::VarOpType::VAR_OP_COS||
                    variableOpParams->varOpType == VariableOpParams::VarOpType::VAR_OP_COSH||
                    variableOpParams->varOpType == VariableOpParams::VarOpType::VAR_OP_SIN||
                    variableOpParams->varOpType == VariableOpParams::VarOpType::VAR_OP_SINH||
                    variableOpParams->varOpType == VariableOpParams::VarOpType::VAR_OP_TAN||
                    variableOpParams->varOpType == VariableOpParams::VarOpType::VAR_OP_TANH||
                    variableOpParams->varOpType == VariableOpParams::VarOpType::VAR_OP_EXP||
                    variableOpParams->varOpType == VariableOpParams::VarOpType::VAR_OP_POW||
                    variableOpParams->varOpType == VariableOpParams::VarOpType::VAR_OP_LOG||
                    variableOpParams->varOpType == VariableOpParams::VarOpType::VAR_OP_LOG10||
                    variableOpParams->varOpType == VariableOpParams::VarOpType::VAR_OP_SQRT
                    )
            {
                if(variableOpParams->layerIndexes.size() !=0)
                {
                    throw Exception(1, "VarOp layer error, should not add extra layer  ", __FILE__, __LINE__, __FUNCTION__);
                }
            }
            else
            {
                throw Exception(1, "VarOp layer error, a op is not supported  ", __FILE__, __LINE__, __FUNCTION__);
            }

            layer                                   =   new VariableOpLayer(params.batch, params.width, params.height, params.channels, variableOpParams->layerIndexes, variableOpParams->varOpType, variableOpParams->constVal);
        }
        else if(_parser->params[i]->type == LayerType::PERMUTE)
        {
            PermuteParams *permuteParams            =   reinterpret_cast<PermuteParams*>(_parser->params[i]);
            layer                                   =   new PermuteLayer(params.batch, params.height, params.width, params.channels, permuteParams->dim0, permuteParams->dim1, permuteParams->dim2);
        }
        else if(_parser->params[i]->type == LayerType::REDUCTION)
        {
            ReductionParams *reductionParams        =   reinterpret_cast<ReductionParams*>(_parser->params[i]);
            layer                                   =   new ReductionLayer(params.batch, params.height, params.width, params.channels, reductionParams->axis, reductionParams->reduceType);
        }
        else if(_parser->params[i]->type == LayerType::UPSAMPLE)
        {
            UpSampleParams *upSampleParams          =   reinterpret_cast<UpSampleParams*>(_parser->params[i]);
            layer                                   =   new UpSampleLayer(params.batch, params.width, params.height, params.channels, upSampleParams->strideX, upSampleParams->strideY,
                                                                          upSampleParams->scaleX, upSampleParams->scaleY, upSampleParams->upsampleType, upSampleParams->alignCorners);
        }
        else if(_parser->params[i]->type == LayerType::SOFTMAX)
        {
            SoftMaxParams  *softmaxParams           =   reinterpret_cast<SoftMaxParams*>(_parser->params[i]);
            layer                                   =   new SoftMaxLayer(params.batch, params.inputNums, softmaxParams->groups, softmaxParams->temperature);
        }
        else if(_parser->params[i]->type == LayerType::YOLOV3)
        {
            Yolov3Params *yolov3Params              =   reinterpret_cast<Yolov3Params*>(_parser->params[i]);
            layer                                   =   new Yolov3Layer(params.batch, params.width, params.height, params.channels, yolov3Params->orgWidth, yolov3Params->orgHeight,
                                                                        yolov3Params->classNum, yolov3Params->anchors);
        }
        else if(_parser->params[i]->type == LayerType::YOLOV3_OUT)
        {
            Yolov3OutParams     *yolov3OutParams    =   reinterpret_cast<Yolov3OutParams*>(_parser->params[i]);

            std::vector<Yolov3Info> yolov3LayersInfo;

            if(yolov3OutParams->layerIndexes.size() < 1)
            {
                throw Exception(1, "yolov3out layer error, no yolov3 layers", __FILE__, __LINE__, __FUNCTION__);
            }

            for (size_t k = 0; k < yolov3OutParams->layerIndexes.size(); ++k)
            {
                size_t index   =   static_cast<size_t>(yolov3OutParams->layerIndexes[k]);

                if(_net->layers[index]->type() != LayerType::YOLOV3)
                {
                    throw Exception(1, "yolov3out layer error, not a yolov3 layer", __FILE__, __LINE__, __FUNCTION__);
                }

                yolov3LayersInfo.push_back(Yolov3Info(_net->layers[index]->getOutHeight(),
                                                      _net->layers[index]->getOutWidth(),
                                                      _net->layers[index]->getOutChannel()
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

        if(layer->getInputSpaceSize() > maxInputSpace)
        {
            maxInputSpace = layer->getInputSpaceSize();
        }
        _net->layers.push_back(layer);
    }
    _netState->workspace     =   new float[maxWorkSpace]();

#ifdef USE_GPU
    if(maxInputSpace != 0)
    {
        _netState->gpuWorkspace  =   Cuda::makeCudaArray(_netState->workspace,maxWorkSpace);
    }
    else
    {
        std::cout<<" Warning: workspace size equal 0 "<<std::endl;
    }

    if(BaseLayer::useFp16)
    {
        _netState->gpuInputFp16 = (float*)Cuda::makeFp16ArrayFromFp32(nullptr, maxInputSpace);
    }
#endif

}

void NetBuilder::loadWeightsFromMsnhBin(const string &path)
{
    if(BaseLayer::isPreviewMode)
    {
        throw Exception(1, "Can not load weights in preview mode !",__FILE__, __LINE__, __FUNCTION__);
    }

    _parser->readMsnhBin(path);
    size_t ptr = 0;
    std::vector<float>::const_iterator first = _parser->msnhF32Weights.begin();

    for (size_t i = 0; i < _net->layers.size(); ++i)
    {
        if(_net->layers[i]->type() == LayerType::CONVOLUTIONAL || _net->layers[i]->type() == LayerType::CONNECTED || _net->layers[i]->type() == LayerType::BATCHNORM ||
                _net->layers[i]->type() == LayerType::RES_BLOCK   || _net->layers[i]->type() == LayerType::RES_2_BLOCK || _net->layers[i]->type() == LayerType::ADD_BLOCK ||
                _net->layers[i]->type() == LayerType::CONCAT_BLOCK  || _net->layers[i]->type() == LayerType::DECONVOLUTIONAL)
        {
            size_t nums = _net->layers[i]->getNumWeights();

            if((ptr + nums) > (_parser->msnhF32Weights.size()))
            {
                throw Exception(1,"Load weights err, need > given. Needed :" + std::to_string(ptr + nums) + " given :" +
                                std::to_string(_parser->msnhF32Weights.size()),__FILE__,__LINE__, __FUNCTION__);
            }

            std::vector<float> weights(first +  static_cast<long long>(ptr), first + static_cast<long long>(ptr + nums));

            _net->layers[i]->loadAllWeigths(weights);

            ptr         =   ptr + nums;
        }
    }

    if(ptr != _parser->msnhF32Weights.size())
    {
        throw Exception(1,"Load weights err, need != given. Needed :" + std::to_string(ptr) + " given :" +
                        std::to_string(_parser->msnhF32Weights.size()),__FILE__,__LINE__, __FUNCTION__);
    }

}

void NetBuilder::setPreviewMode(const bool &mode)
{
    BaseLayer::setPreviewMode(mode);
}

void NetBuilder::setForceUseCuda(const bool &onlyUseCuda)
{
    BaseLayer::setForceUseCuda(onlyUseCuda);
}

void NetBuilder::setUseFp16(const bool &useFp16)
{
    BaseLayer::setUseFp16(useFp16);
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
    _netState->input     =   img.data();
    _netState->inputNum  =   static_cast<int>(img.size());
    if(_net->layers[0]->getInputNum() != _netState->inputNum)
    {
        throw Exception(1,"input image size err. Needed :" + std::to_string(_net->layers[0]->getInputNum()) + " given :" +
                std::to_string(img.size()),__FILE__,__LINE__, __FUNCTION__);
    }

    for (size_t i = 0; i < _net->layers.size(); ++i)
    {
        _net->layers[i]->forward(*_netState);

        _netState->input     =   _net->layers[i]->getOutput();
        _netState->inputNum  =   _net->layers[i]->getOutputNum();

        if(i == _net->layers.size()-1)
        {
            this->_lastLayerOutHeight  = _net->layers[i]->getOutHeight();
            this->_lastLayerOutWidth   = _net->layers[i]->getOutWidth();
            this->_lastLayerOutChannel = _net->layers[i]->getOutChannel();
            this->_lastLayerOutNum     = _net->layers[i]->getOutputNum();
        }

    }

#ifdef USE_NNPACK
    nnp_deinitialize();
#endif
    std::vector<float> pred(_netState->input, _netState->input + _netState->inputNum);

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
    _netState->input     =   img.data();
    _netState->inputNum  =   static_cast<int>(img.size());
    if(_net->layers[0]->getInputNum() != _netState->inputNum)
    {
        throw Exception(1,"input image size err. Needed :" + std::to_string(_net->layers[0]->getInputNum()) + " given :" +
                std::to_string(img.size()),__FILE__,__LINE__, __FUNCTION__);
    }

    for (size_t i = 0; i < _net->layers.size(); ++i)
    {

        if(_net->layers[i]->type() != LayerType::ROUTE && _net->layers[i]->type() != LayerType::YOLOV3_OUT) 

        {
            if(_netState->inputNum != _net->layers[i]->getInputNum())
            {
                throw Exception(1, "layer " + to_string(i) + " inputNum needed : " + std::to_string(_net->layers[i]->getInputNum()) +
                                ", given : " + std::to_string(_netState->inputNum),__FILE__,__LINE__, __FUNCTION__);
            }
        }

        _net->layers[i]->forward(*_netState);

        _netState->input     =   _net->layers[i]->getOutput();
        _netState->inputNum  =   _net->layers[i]->getOutputNum();

    }

#ifdef USE_NNPACK
    nnp_deinitialize();
#endif

    if((_net->layers[_net->layers.size()-1])->type() == LayerType::YOLOV3_OUT)
    {
        return (reinterpret_cast<Yolov3OutLayer*>((_net->layers[_net->layers.size()-1])))->finalOut;
    }
    else
    {
        throw Exception(1,"not a yolov3 net", __FILE__, __LINE__, __FUNCTION__);
    }
}

#ifdef USE_GPU
std::vector<float> NetBuilder::runClassifyGPU(std::vector<float> img)
{
    _gpuInferenceTime = 0;
    auto st = TimeUtil::startRecord();
    if(BaseLayer::isPreviewMode)
    {
        throw Exception(1, "Can not infer in preview mode !",__FILE__, __LINE__, __FUNCTION__);
    }

    _netState->inputNum  =   static_cast<int>(img.size());

    if(_net->layers[0]->getInputNum() != _netState->inputNum)
    {
        throw Exception(1,"input image size err. Needed :" + std::to_string(_net->layers[0]->getInputNum()) + " given :" +
                std::to_string(img.size()),__FILE__,__LINE__, __FUNCTION__);
    }

    _netState->input     =   Cuda::makeCudaArray(img.data(), img.size());

    for (size_t i = 0; i < _net->layers.size(); ++i)
    {
        _net->layers[i]->forwardGPU(*_netState);
        _netState->input     =   _net->layers[i]->getGpuOutput();
        _netState->inputNum  =   _net->layers[i]->getOutputNum();

        if(i == _net->layers.size()-1)
        {
            this->_lastLayerOutHeight  = _net->layers[i]->getOutHeight();
            this->_lastLayerOutWidth   = _net->layers[i]->getOutWidth();
            this->_lastLayerOutChannel = _net->layers[i]->getOutChannel();
            this->_lastLayerOutNum     = _net->layers[i]->getOutputNum();
        }
    }

    float* out = new float[_netState->inputNum]();
    Cuda::pullCudaArray(_netState->input, out,_netState->inputNum);
    std::vector<float> pred(out, out + _netState->inputNum);
    delete[] out;
    out = nullptr;

    _gpuInferenceTime = TimeUtil::getElapsedTime(st);
    return pred;
}

std::vector<std::vector<Yolov3Box>> NetBuilder::runYolov3GPU(std::vector<float> img)
{
    _gpuInferenceTime = 0;
    auto st = TimeUtil::startRecord();
    if(BaseLayer::isPreviewMode)
    {
        throw Exception(1, "Can not infer in preview mode !",__FILE__, __LINE__, __FUNCTION__);
    }
    _netState->inputNum  =   static_cast<int>(img.size());

    if(_net->layers[0]->getInputNum() != _netState->inputNum)
    {
        throw Exception(1,"input image size err. Needed :" + std::to_string(_net->layers[0]->getInputNum()) + " given :" +
                std::to_string(img.size()),__FILE__,__LINE__, __FUNCTION__);
    }

    _netState->input     =   Cuda::makeCudaArray(img.data(), img.size());

    for (size_t i = 0; i < _net->layers.size(); ++i)
    {

        if(_net->layers[i]->type() != LayerType::ROUTE && _net->layers[i]->type() != LayerType::YOLOV3_OUT) 

        {
            if(_netState->inputNum != _net->layers[i]->getInputNum())
            {
                throw Exception(1, "layer " + to_string(i) + " inputNum needed : " + std::to_string(_net->layers[i]->getInputNum()) +
                                ", given : " + std::to_string(_netState->inputNum),__FILE__,__LINE__, __FUNCTION__);
            }
        }

        _net->layers[i]->forwardGPU(*_netState);

        _netState->input     =   _net->layers[i]->getGpuOutput();
        _netState->inputNum  =   _net->layers[i]->getOutputNum();

    }

    if((_net->layers[_net->layers.size()-1])->type() == LayerType::YOLOV3_OUT)
    {
        _gpuInferenceTime = TimeUtil::getElapsedTime(st);
        return (reinterpret_cast<Yolov3OutLayer*>((_net->layers[_net->layers.size()-1])))->finalOut;
    }
    else
    {
        throw Exception(1,"not a yolov3 net", __FILE__, __LINE__, __FUNCTION__);
    }
}

#endif

Point2I NetBuilder::getInputSize()
{
    if(_parser->params.empty())
    {
        throw Exception(1,"net param is empty", __FILE__, __LINE__, __FUNCTION__);
    }

    if(_parser->params[0]->type == LayerType::CONFIG)
    {
        NetConfigParams* params = reinterpret_cast<NetConfigParams*>(_parser->params[0]);
        return Point2I(params->width,params->height);
    }
    else
    {
        throw Exception(1,"net param error", __FILE__, __LINE__, __FUNCTION__);
    }
}

int NetBuilder::getInputChannel()
{
    if(_parser->params.empty())
    {
        throw Exception(1,"net param is empty", __FILE__, __LINE__, __FUNCTION__);
    }

    if(_parser->params[0]->type == LayerType::CONFIG)
    {
        NetConfigParams* params = reinterpret_cast<NetConfigParams*>(_parser->params[0]);
        return params->channels;
    }
    else
    {
        throw Exception(1,"net param error", __FILE__, __LINE__, __FUNCTION__);
    }
}

void NetBuilder::clearLayers()
{
    for (size_t i = 0; i < _net->layers.size(); ++i)
    {
        if(_net->layers[i]!=nullptr)
        {
            if(_net->layers[i]->type() == LayerType::CONVOLUTIONAL)
            {
                delete reinterpret_cast<ConvolutionalLayer*>(_net->layers[i]);
            }
            else if(_net->layers[i]->type() == LayerType::MAXPOOL)
            {
                delete reinterpret_cast<MaxPoolLayer*>(_net->layers[i]);
            }
            else if(_net->layers[i]->type() == LayerType::ACTIVE)
            {
                delete reinterpret_cast<ActivationLayer*>(_net->layers[i]);
            }
            else if(_net->layers[i]->type() == LayerType::SOFTMAX)
            {
                delete reinterpret_cast<SoftMaxLayer*>(_net->layers[i]);
            }
            else if(_net->layers[i]->type() == LayerType::CONNECTED)
            {
                delete reinterpret_cast<ConnectedLayer*>(_net->layers[i]);
            }
            else if(_net->layers[i]->type() == LayerType::BATCHNORM)
            {
                delete reinterpret_cast<BatchNormLayer*>(_net->layers[i]);
            }
            else if(_net->layers[i]->type() == LayerType::LOCAL_AVGPOOL)
            {
                delete reinterpret_cast<LocalAvgPoolLayer*>(_net->layers[i]);
            }
            else if(_net->layers[i]->type() == LayerType::GLOBAL_AVGPOOL)
            {
                delete reinterpret_cast<GlobalAvgPoolLayer*>(_net->layers[i]);
            }
            else if(_net->layers[i]->type() == LayerType::RES_BLOCK)
            {
                delete reinterpret_cast<ResBlockLayer*>(_net->layers[i]);
            }
            else if(_net->layers[i]->type() == LayerType::RES_2_BLOCK)
            {
                delete reinterpret_cast<Res2BlockLayer*>(_net->layers[i]);
            }
            else if(_net->layers[i]->type() == LayerType::ADD_BLOCK)
            {
                delete reinterpret_cast<AddBlockLayer*>(_net->layers[i]);
            }
            else if(_net->layers[i]->type() == LayerType::CONCAT_BLOCK)
            {
                delete reinterpret_cast<ConcatBlockLayer*>(_net->layers[i]);
            }
            else if(_net->layers[i]->type() == LayerType::ROUTE)
            {
                delete reinterpret_cast<RouteLayer*>(_net->layers[i]);
            }
            else if(_net->layers[i]->type() == LayerType::VARIABLE_OP)
            {
                delete reinterpret_cast<VariableOpLayer*>(_net->layers[i]);
            }
            else if(_net->layers[i]->type() == LayerType::UPSAMPLE)
            {
                delete reinterpret_cast<UpSampleLayer*>(_net->layers[i]);
            }
            else if(_net->layers[i]->type() == LayerType::YOLOV3)
            {
                delete reinterpret_cast<Yolov3Layer*>(_net->layers[i]);
            }
            else if(_net->layers[i]->type() == LayerType::YOLOV3_OUT)
            {
                delete reinterpret_cast<Yolov3OutLayer*>(_net->layers[i]);
            }
            else if(_net->layers[i]->type() == LayerType::PADDING)
            {
                delete reinterpret_cast<PaddingLayer*>(_net->layers[i]);
            }
            else if(_net->layers[i]->type() == LayerType::PERMUTE)
            {
                delete reinterpret_cast<PermuteLayer*>(_net->layers[i]);
            }
            else if(_net->layers[i]->type() == LayerType::REDUCTION)
            {
                delete reinterpret_cast<ReductionLayer*>(_net->layers[i]);
            }
            _net->layers[i] = nullptr;
        }

        if(i == (_net->layers.size()-1))
        {
            _net->layers.clear();
        }
    }
}

float NetBuilder::getInferenceTime()
{
    float inferTime     =   0.f;
    for (size_t i = 0; i < this->_net->layers.size(); ++i)
    {
        inferTime       +=  this->_net->layers[i]->getForwardTime();
    }

    return inferTime;
}

string NetBuilder::getLayerDetail()
{
    std::string detail;
    for(size_t i=0;i<this->_net->layers.size();++i)
    {
        detail = detail + "────────────────────────────────  " + ((i<10)?("00"+std::to_string(i)):((i<100)?("0"+std::to_string(i)):std::to_string(i)))
                + " ─────────────────────────────────\n";
        detail = detail + this->_net->layers[i]->getLayerDetail()  + "weights : "+
                std::to_string(this->_net->layers[i]->getNumWeights()) + "\n\n";

    }
    return detail;
}

string NetBuilder::getTimeDetail()
{
    float totalWaste = getInferenceTime();
    std::string detail;
    detail     = detail + "LAYER            INDEX       TIME         LAYER_t/TOTAL_t\r\n"
            + "=========================================================\n";

    for(size_t i=0;i<this->_net->layers.size();++i)
    {
        detail = detail + this->_net->layers[i]->getLayerName() + " : ";
        detail = detail + ((i<10)?("00"+std::to_string(i)):((i<100)?("0"+std::to_string(i)):std::to_string(i))) + "     ";
        detail = detail + Msnhnet::ExString::left(std::to_string(this->_net->layers[i]->getForwardTime()),6) +" ms         ";
        detail = detail + Msnhnet::ExString::left(std::to_string(((int)(this->_net->layers[i]->getForwardTime() / totalWaste *1000))/10.f),4) + "%\n";
    }
    detail     = detail + "=========================================================\n";
    detail     = detail + "Msnhnet inference time : " + std::to_string(totalWaste) + " ms";
    return detail;
}

float NetBuilder::getGpuInferenceTime() const
{
    return _gpuInferenceTime;
}

Network *NetBuilder::getNet() const
{
    return _net;
}

int NetBuilder::getLastLayerOutWidth() const
{
    return _lastLayerOutWidth;
}

int NetBuilder::getLastLayerOutHeight() const
{
    return _lastLayerOutHeight;
}

int NetBuilder::getLastLayerOutChannel() const
{
    return _lastLayerOutChannel;
}

size_t NetBuilder::getLastLayerOutNum() const
{
    return _lastLayerOutNum;
}

}
