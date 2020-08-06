#include "Msnhnet/io/MsnhParser.h"
namespace Msnhnet
{
int BaseParams::index  = -1;

Parser::Parser()
{

}

Parser::~Parser()
{
    clearParams();
}

void Parser::clearParams()
{
    BaseParams::index = 0;

    for (size_t i = 0; i < params.size(); ++i)
    {
        if(params[i]!=nullptr)
        {

            if(params[i]->type == LayerType::CONFIG)
            {
                delete reinterpret_cast<NetConfigParams*>(params[i]);

            }
            else if(params[i]->type == LayerType::CONVOLUTIONAL)
            {
                delete reinterpret_cast<ConvParams*>(params[i]);
            }
            else if(params[i]->type == LayerType::ACTIVE)
            {
                delete reinterpret_cast<ActivationParams*>(params[i]);
            }
            else if(params[i]->type == LayerType::SOFTMAX)
            {
                delete reinterpret_cast<SoftMaxParams*>(params[i]);
            }
            else if(params[i]->type == LayerType::EMPTY)
            {
                delete reinterpret_cast<EmptyParams*>(params[i]);
            }
            else if(params[i]->type == LayerType::PADDING)
            {
                delete reinterpret_cast<PaddingParams*>(params[i]);
            }
            else if(params[i]->type == LayerType::MAXPOOL)
            {
                delete reinterpret_cast<MaxPoolParams*>(params[i]);
            }
            else if(params[i]->type == LayerType::LOCAL_AVGPOOL)
            {
                delete reinterpret_cast<LocalAvgPoolParams*>(params[i]);
            }
            else if(params[i]->type == LayerType::CONNECTED)
            {
                delete reinterpret_cast<ConnectParams*>(params[i]);
            }
            else if(params[i]->type == LayerType::BATCHNORM)
            {
                delete reinterpret_cast<BatchNormParams*>(params[i]);
            }
            else if(params[i]->type == LayerType::RES_BLOCK)
            {
                delete reinterpret_cast<ResBlockParams*>(params[i]);
            }
            else if(params[i]->type == LayerType::RES_2_BLOCK)
            {
                delete reinterpret_cast<Res2BlockParams*>(params[i]);
            }
            else if(params[i]->type == LayerType::CONCAT_BLOCK)
            {
                delete reinterpret_cast<ConcatBlockParams*>(params[i]);
            }
            else if(params[i]->type == LayerType::ADD_BLOCK)
            {
                delete reinterpret_cast<AddBlockParams*>(params[i]);
            }
            else if(params[i]->type == LayerType::ROUTE)
            {
                delete reinterpret_cast<RouteParams*>(params[i]);
            }
            else if(params[i]->type == LayerType::UPSAMPLE)
            {
                delete reinterpret_cast<UpSampleParams*>(params[i]);
            }
            else if(params[i]->type == LayerType::YOLOV3)
            {
                delete reinterpret_cast<Yolov3Params*>(params[i]);
            }
            else if(params[i]->type == LayerType::YOLOV3_OUT)
            {
                delete reinterpret_cast<Yolov3OutParams*>(params[i]);
            }
            params[i] = nullptr;
        }

        if(i == (params.size()-1))
        {
            params.clear();
        }
    }
}

void Parser::readCfg(const std::string &path)
{
    clearParams();

    try
    {
        YAML::Node root = YAML::LoadFile(path);

        int index = 0;

        for (YAML::const_iterator it = root.begin(); it != root.end(); ++it)
        {
            index++;

            std::string node = it->first.as<std::string>();

            if(index == 1)
            {
                if(node == "config")
                {
                    if(it->second.Type() == YAML::NodeType::Map)
                    {
                        NetConfigParams *netConfigParams = new NetConfigParams(false); 

                        parseConfigParams(netConfigParams, it);
                        params.push_back(netConfigParams);
                        continue;
                    }
                    else
                    {
                        throw Exception(1,"[config] content error", __FILE__, __LINE__, __FUNCTION__);
                    }
                }
                else
                {
                    throw Exception(1,"first node must be [config]", __FILE__, __LINE__, __FUNCTION__);
                }
            }

            if(node == "maxpool")
            {
                if(it->second.Type() == YAML::NodeType::Map)
                {
                    MaxPoolParams *maxPoolParams = new MaxPoolParams(true);
                    parseMaxPoolParams(maxPoolParams, it);
                    params.push_back(maxPoolParams);
                }
                else
                {
                    throw Exception(1,"[maxpool] content error", __FILE__, __LINE__, __FUNCTION__);
                }
            }
            else if(node == "act")
            {
                if(it->second.Type() == YAML::NodeType::Map)
                {
                    ActivationParams *activationParams = new ActivationParams(true);
                    parseActivationParams(activationParams, it);
                    params.push_back(activationParams);
                }
                else
                {
                    throw Exception(1,"[act] content error", __FILE__, __LINE__, __FUNCTION__);
                }
            }
            else if(node == "padding")
            {
                if(it->second.Type() == YAML::NodeType::Map)
                {
                    PaddingParams *paddingParams = new PaddingParams(true);
                    parsePaddingParams(paddingParams, it);
                    params.push_back(paddingParams);
                }
                else
                {
                    throw Exception(1,"[padding] content error", __FILE__, __LINE__, __FUNCTION__);
                }
            }
            else if(node == "localavgpool")
            {
                if(it->second.Type() == YAML::NodeType::Map)
                {
                    LocalAvgPoolParams *localAvgPoolParams = new LocalAvgPoolParams(true);
                    parseLocalAvgPoolParams(localAvgPoolParams, it);
                    params.push_back(localAvgPoolParams);
                }
                else
                {
                    throw Exception(1,"[localavgpool] content error", __FILE__, __LINE__, __FUNCTION__);
                }
            }
            else if(node == "globalavgpool")
            {
                GlobalAvgPoolParams *globalAvgPoolParams = new GlobalAvgPoolParams(true);
                params.push_back(globalAvgPoolParams);
            }
            else if(node == "conv")
            {
                if(it->second.Type() == YAML::NodeType::Map)
                {
                    ConvParams *convParams = new ConvParams(true);
                    parseConvParams(convParams, it);
                    params.push_back(convParams);
                }
                else
                {
                    throw Exception(1,"[conv] content error", __FILE__, __LINE__, __FUNCTION__);
                }
            }
            else if(node == "deconv")
            {
                if(it->second.Type() == YAML::NodeType::Map)
                {
                    DeConvParams *deconvParams = new DeConvParams(true);
                    parseDeConvParams(deconvParams, it);
                    params.push_back(deconvParams);
                }
                else
                {
                    throw Exception(1,"[conv] content error", __FILE__, __LINE__, __FUNCTION__);
                }
            }
            else if(node == "connect")
            {
                if(it->second.Type() == YAML::NodeType::Map)
                {
                    ConnectParams *connectParams = new ConnectParams(true);
                    parseConnectParams(connectParams, it);
                    params.push_back(connectParams);
                }
                else
                {
                    throw Exception(1,"[connect] content error", __FILE__, __LINE__, __FUNCTION__);
                }
            }
            else if(node == "batchnorm")
            {
                if(it->second.Type() == YAML::NodeType::Map)
                {
                    BatchNormParams *batchNormParams = new BatchNormParams(true);
                    parseBatchNormParams(batchNormParams, it);
                    params.push_back(batchNormParams);
                }
                else
                {
                    throw Exception(1,"[batchnorm] content error", __FILE__, __LINE__, __FUNCTION__);
                }
            }
            else if(node == "resblock")
            {
                if(it->second.Type() == YAML::NodeType::Map)
                {
                    int size = 1;

                    ActivationType act = ActivationType::NONE;
                    std::vector<float> tmpActParams;

                    for (YAML::const_iterator it2 = it->second.begin(); it2 != it->second.end(); ++it2)
                    {
                        std::string key     =   it2->first.as<std::string>();

                        if(key == "size")
                        {
                            std::string value   =   it2->second.as<std::string>(); 

                            if(!ExString::strToInt(value, size))
                            {
                                throw Exception(1,"[resblock] size can't convert to int", __FILE__, __LINE__, __FUNCTION__);
                            }
                        }

                        if(key == "activation")
                        {
                            std::string value   =   it2->second.as<std::string>(); 

                            std::vector<std::string> splits;
                            ExString::split(splits, value, ",");
                            act = Activations::getActivation(splits[0]);

                            if(splits.size()>1)
                            {
                                for (size_t i = 1; i < splits.size(); ++i)
                                {
                                    float tmp = 0.f;
                                    ExString::strToFloat(splits[i], tmp);
                                    tmpActParams.push_back(tmp);
                                }
                            }
                        }
                    }

                    for (int i = 0; i < size; ++i)
                    {

                        ResBlockParams *resBlockParams = new ResBlockParams(true);
                        parseResBlockParams(resBlockParams, it);
                        resBlockParams->activation = act;
                        params.push_back(resBlockParams);
                    }
                }
                else
                {
                    throw Exception(1,"[resblock] content error", __FILE__, __LINE__, __FUNCTION__);
                }
            }
            else if(node == "res2block")
            {
                if(it->second.Type() == YAML::NodeType::Map)
                {
                    int size = 1;

                    ActivationType act = ActivationType::NONE;
                    std::vector<float> tmpActParams;

                    for (YAML::const_iterator it2 = it->second.begin(); it2 != it->second.end(); ++it2)
                    {
                        std::string key     =   it2->first.as<std::string>();

                        if(key == "size")
                        {
                            std::string value   =   it2->second.as<std::string>(); 

                            if(!ExString::strToInt(value, size))
                            {
                                throw Exception(1,"[res2block] size can't convert to int", __FILE__, __LINE__, __FUNCTION__);
                            }
                        }

                        if(key == "activation")
                        {
                            std::string value   =   it2->second.as<std::string>(); 

                            std::vector<std::string> splits;
                            ExString::split(splits, value, ",");
                            act = Activations::getActivation(splits[0]);

                            if(splits.size()>1)
                            {
                                for (size_t i = 1; i < splits.size(); ++i)
                                {
                                    float tmp = 0.f;
                                    ExString::strToFloat(splits[i], tmp);
                                    tmpActParams.push_back(tmp);
                                }
                            }
                        }
                    }

                    for (int i = 0; i < size; ++i)
                    {
                        Res2BlockParams *res2BlockParams  = new Res2BlockParams(true);
                        parseRes2BlockParams(res2BlockParams, it);
                        res2BlockParams->activation       = act;
                        res2BlockParams->actParams         = tmpActParams;
                        params.push_back(res2BlockParams);
                    }
                }
                else
                {
                    throw Exception(1,"[res2block] content error", __FILE__, __LINE__, __FUNCTION__);
                }
            }
            else if(node == "addblock")
            {
                if(it->second.Type() == YAML::NodeType::Map)
                {
                    int size = 1;

                    ActivationType act = ActivationType::NONE;
                    std::vector<float> tmpActParams;

                    for (YAML::const_iterator it2 = it->second.begin(); it2 != it->second.end(); ++it2)
                    {
                        std::string key     =   it2->first.as<std::string>();

                        if(key == "size")
                        {
                            std::string value   =   it2->second.as<std::string>(); 

                            if(!ExString::strToInt(value, size))
                            {
                                throw Exception(1,"[addblock] size can't convert to int", __FILE__, __LINE__, __FUNCTION__);
                            }
                        }

                        if(key == "activation")
                        {
                            std::string value   =   it2->second.as<std::string>(); 

                            std::vector<std::string> splits;
                            ExString::split(splits, value, ",");
                            act = Activations::getActivation(splits[0]);

                            if(splits.size()>1)
                            {
                                for (size_t i = 1; i < splits.size(); ++i)
                                {
                                    float tmp = 0.f;
                                    ExString::strToFloat(splits[i], tmp);
                                    tmpActParams.push_back(tmp);
                                }
                            }
                        }
                    }

                    for (int i = 0; i < size; ++i)
                    {
                        AddBlockParams *addBlockParams  = new AddBlockParams(true);
                        parseAddBlockParams(addBlockParams, it);
                        addBlockParams->activation      = act;
                        addBlockParams->actParams       = tmpActParams;
                        params.push_back(addBlockParams);
                    }
                }
            }
            else if(node == "concatblock")
            {
                if(it->second.Type() == YAML::NodeType::Map)
                {
                    int size = 1;

                    ActivationType act = ActivationType::NONE;
                    std::vector<float> tmpActParams;

                    for (YAML::const_iterator it2 = it->second.begin(); it2 != it->second.end(); ++it2)
                    {
                        std::string key     =   it2->first.as<std::string>();

                        if(key == "size")
                        {
                            std::string value   =   it2->second.as<std::string>(); 

                            if(!ExString::strToInt(value, size))
                            {
                                throw Exception(1,"[concatblock] size can't convert to int", __FILE__, __LINE__, __FUNCTION__);
                            }
                        }

                        if(key == "activation")
                        {
                            std::string value   =   it2->second.as<std::string>(); 

                            std::vector<std::string> splits;
                            ExString::split(splits, value, ",");
                            act = Activations::getActivation(splits[0]);

                            if(splits.size()>1)
                            {
                                for (size_t i = 1; i < splits.size(); ++i)
                                {
                                    float tmp = 0.f;
                                    ExString::strToFloat(splits[i], tmp);
                                    tmpActParams.push_back(tmp);
                                }
                            }
                        }
                    }

                    for (int i = 0; i < size; ++i)
                    {
                        ConcatBlockParams *concatBlockParams  = new ConcatBlockParams(true);
                        parseConcatBlockParams(concatBlockParams, it);
                        concatBlockParams->activation         = act;
                        concatBlockParams->actParams          = tmpActParams;
                        params.push_back(concatBlockParams);
                    }
                }
                else
                {
                    throw Exception(1,"[addblock] content error", __FILE__, __LINE__, __FUNCTION__);
                }
            }
            else if(node == "route")
            {
                if(it->second.Type() == YAML::NodeType::Map)
                {
                    RouteParams *routeParams = new RouteParams(true);
                    parseRouteParams(routeParams, it);
                    params.push_back(routeParams);
                }
                else
                {
                    throw Exception(1,"[route] content error", __FILE__, __LINE__, __FUNCTION__);
                }
            }
            else if(node == "softmax")
            {
                if(it->second.Type() == YAML::NodeType::Map)
                {
                    SoftMaxParams *softMaxParams = new SoftMaxParams(true);
                    parseSoftMaxParams(softMaxParams, it);
                    params.push_back(softMaxParams);
                }
                else
                {
                    throw Exception(1,"[softmax] content error", __FILE__, __LINE__, __FUNCTION__);
                }
            }
            else if(node == "upsample")
            {
                if(it->second.Type() == YAML::NodeType::Map)
                {
                    UpSampleParams *upSampleParams = new UpSampleParams(true);
                    parseUpSampleParams(upSampleParams, it);
                    params.push_back(upSampleParams);
                }
                else
                {
                    throw Exception(1,"[upsample] content error", __FILE__, __LINE__, __FUNCTION__);
                }
            }
            else if(node == "yolov3")
            {
                if(it->second.Type() == YAML::NodeType::Map)
                {
                    Yolov3Params *yolov3Params  = new Yolov3Params(true);
                    parseYolov3Params(yolov3Params, it);
                    yolov3Params->orgHeight     =  (reinterpret_cast<NetConfigParams*>(params[0]))->height;
                    yolov3Params->orgWidth      =  (reinterpret_cast<NetConfigParams*>(params[0]))->width;
                    params.push_back(yolov3Params);
                }
                else
                {
                    throw Exception(1,"[yolov3] content error", __FILE__, __LINE__, __FUNCTION__);
                }
            }
            else if(node == "yolov3out")
            {
                if(it->second.Type() == YAML::NodeType::Map)
                {
                    Yolov3OutParams *yolov3OutParams = new Yolov3OutParams(true);
                    parseYolov3OutParams(yolov3OutParams, it);
                    yolov3OutParams->orgHeight       =  (reinterpret_cast<NetConfigParams*>(params[0]))->height;
                    yolov3OutParams->orgWidth        =  (reinterpret_cast<NetConfigParams*>(params[0]))->width;
                    params.push_back(yolov3OutParams);
                }
                else
                {
                    throw Exception(1,"[yolov3] content error", __FILE__, __LINE__, __FUNCTION__);
                }
            }
            else
            {
                throw Exception(1,"LayerType [" + node + "] is not supported", __FILE__, __LINE__, __FUNCTION__);
            }
        }
    }
    catch(YAML::Exception ex)
    {
        throw Exception(1,ex.what(), __FILE__, __LINE__, __FUNCTION__);
    }
    return;
}

void Parser::readMsnhBin(const std::string &path)
{
    std::ifstream readFile;
    readFile.open(path,std::ios::in|std::ios::binary);

    if(!readFile.is_open())
    {
        throw Exception(0,std::string(path) + " open filed!", __FILE__, __LINE__, __FUNCTION__);
    }

    readFile.seekg(0, std::ios::end);
    auto fsize = readFile.tellg();
    readFile.seekg(0, std::ios::beg);

    if (fsize < 1)
    {
        throw Exception(0,std::string(path) + " read filed!", __FILE__, __LINE__, __FUNCTION__);
    }

    if (fsize % 4 != 0)
    {
        throw Exception(0,std::string(path) + " file error!", __FILE__, __LINE__, __FUNCTION__);
    }

    char *data = new char[static_cast<size_t>(fsize)]();
    readFile.read(data, fsize);

    Float32 float32;

    msnhF32Weights.clear();

    for (int i = 0;  i < fsize; i+=4)
    {
        float32.bytes[0] = static_cast<uint8_t>(data[i + 0]);
        float32.bytes[1] = static_cast<uint8_t>(data[i + 1]);
        float32.bytes[2] = static_cast<uint8_t>(data[i + 2]);
        float32.bytes[3] = static_cast<uint8_t>(data[i + 3]);
        msnhF32Weights.push_back(float32.val);
    }

    delete[] data;
    data = nullptr;
}

void Parser::parseConfigParams(NetConfigParams *netConfigParams, YAML::const_iterator &iter)
{
    for (YAML::const_iterator it = iter->second.begin(); it != iter->second.end(); ++it)
    {
        std::string key         =   it->first.as<std::string>();
        std::string value       =   it->second.as<std::string>();

        if(key == "batch")
        {
            if(!ExString::strToInt(value, netConfigParams->batch))
            {
                throw Exception(1,"[config] width can't convert to int", __FILE__, __LINE__, __FUNCTION__);
            }
        }
        else if(key == "width")
        {
            if(!ExString::strToInt(value, netConfigParams->width))
            {
                throw Exception(1,"[config] width can't convert to int", __FILE__, __LINE__, __FUNCTION__);
            }
        }
        else if(key == "height")
        {
            if(!ExString::strToInt(value, netConfigParams->height))
            {
                throw Exception(1,"[config] height can't convert to int", __FILE__, __LINE__, __FUNCTION__);
            }
        }
        else if(key == "channels")
        {
            if(!ExString::strToInt(value, netConfigParams->channels))
            {
                throw Exception(1,"[config] channels can't convert to int", __FILE__, __LINE__, __FUNCTION__);
            }
        }
        else
        {
            throw Exception(1, key + " is not supported in [config]", __FILE__, __LINE__, __FUNCTION__);
        }
    }
}

void Parser::parseActivationParams(ActivationParams *activationParams, YAML::const_iterator &iter)
{
    for (YAML::const_iterator it = iter->second.begin(); it != iter->second.end(); ++it)
    {
        std::string key         =   it->first.as<std::string>();
        std::string value       =   it->second.as<std::string>();

        if(key == "activation")
        {
            activationParams->activation = Activations::getActivation(value);
        }
        else
        {
            throw Exception(1, key + " is not supported in [act]", __FILE__, __LINE__, __FUNCTION__);
        }
    }
}

void Parser::parseMaxPoolParams(MaxPoolParams *maxPoolParams, YAML::const_iterator &iter)
{
    for (YAML::const_iterator it = iter->second.begin(); it != iter->second.end(); ++it)
    {

        std::string key     =   it->first.as<std::string>();
        std::string value   =   it->second.as<std::string>();

        if(key == "kSize")
        {
            if(!ExString::strToInt(value, maxPoolParams->kSize))
            {
                throw Exception(1,"[maxpool] kSize can't convert to int", __FILE__, __LINE__, __FUNCTION__);
            }
        }
        else if(key == "kSizeX")
        {
            if(!ExString::strToInt(value, maxPoolParams->kSizeX))
            {
                throw Exception(1,"[maxpool] kSizeX can't convert to int", __FILE__, __LINE__, __FUNCTION__);
            }
        }
        else if(key == "kSizeY")
        {
            if(!ExString::strToInt(value, maxPoolParams->kSizeY))
            {
                throw Exception(1,"[maxpool] kSizeY can't convert to int", __FILE__, __LINE__, __FUNCTION__);
            }
        }
        else if(key == "stride")
        {
            if(!ExString::strToInt(value, maxPoolParams->stride))
            {
                throw Exception(1,"[maxpool] stride can't convert to int", __FILE__, __LINE__, __FUNCTION__);
            }
        }
        else if(key == "strideX")
        {
            if(!ExString::strToInt(value, maxPoolParams->strideX))
            {
                throw Exception(1,"[maxpool] strideX can't convert to int", __FILE__, __LINE__, __FUNCTION__);
            }
        }
        else if(key == "strideY")
        {
            if(!ExString::strToInt(value, maxPoolParams->strideY))
            {
                throw Exception(1,"[maxpool] strideY can't convert to int", __FILE__, __LINE__, __FUNCTION__);
            }
        }
        else if(key == "maxPoolDepth")
        {
            if(!ExString::strToInt(value, maxPoolParams->maxPoolDepth))
            {
                throw Exception(1,"[maxpool] maxPoolDepth can't convert to int", __FILE__, __LINE__, __FUNCTION__);
            }
        }
        else if(key == "outChannels")
        {
            if(!ExString::strToInt(value, maxPoolParams->outChannels))
            {
                throw Exception(1,"[maxpool] outChannels can't convert to int", __FILE__, __LINE__, __FUNCTION__);
            }
        }
        else if(key == "padding")
        {
            if(!ExString::strToInt(value, maxPoolParams->padding))
            {
                throw Exception(1,"[maxpool] padding can't convert to int", __FILE__, __LINE__, __FUNCTION__);
            }
        }
        else if(key == "paddingX")
        {
            if(!ExString::strToInt(value, maxPoolParams->paddingX))
            {
                throw Exception(1,"[maxpool] paddingX can't convert to int", __FILE__, __LINE__, __FUNCTION__);
            }
        }
        else if(key == "paddingY")
        {
            if(!ExString::strToInt(value, maxPoolParams->paddingY))
            {
                throw Exception(1,"[maxpool] paddingY can't convert to int", __FILE__, __LINE__, __FUNCTION__);
            }
        }
        else if(key == "ceilMode")
        {
            if(!ExString::strToInt(value, maxPoolParams->ceilMode))
            {
                throw Exception(1,"[maxpool] ceilMode can't convert to int", __FILE__, __LINE__, __FUNCTION__);
            }
        }
        else
        {
            throw Exception(1, key + " is not supported in [maxpool]", __FILE__, __LINE__, __FUNCTION__);
        }
    }

    if(maxPoolParams->strideX < 0 || maxPoolParams->strideY < 0)
    {
        if(maxPoolParams->strideX < 0 )
        {
            maxPoolParams->strideX = maxPoolParams->stride;
        }

        if(maxPoolParams->strideY < 0 )
        {
            maxPoolParams->strideY = maxPoolParams->stride;
        }
    }

    if(maxPoolParams->kSizeX < 0 || maxPoolParams->kSizeY < 0)
    {
        if(maxPoolParams->kSizeX < 0 )
        {
            maxPoolParams->kSizeX = maxPoolParams->kSize;
        }

        if(maxPoolParams->kSizeY < 0 )
        {
            maxPoolParams->kSizeY = maxPoolParams->kSize;
        }
    }

    if(maxPoolParams->paddingX < 0 || maxPoolParams->paddingY < 0)
    {
        if(maxPoolParams->paddingX < 0 )
        {
            maxPoolParams->paddingX = maxPoolParams->padding;
        }

        if(maxPoolParams->paddingY < 0 )
        {
            maxPoolParams->paddingY = maxPoolParams->padding;
        }
    }
}

void Parser::parseLocalAvgPoolParams(LocalAvgPoolParams *localAvgPoolParams, YAML::const_iterator &iter)
{
    for (YAML::const_iterator it = iter->second.begin(); it != iter->second.end(); ++it)
    {
        std::string key     =   it->first.as<std::string>();
        std::string value   =   it->second.as<std::string>();

        if(key == "kSize")
        {
            if(!ExString::strToInt(value, localAvgPoolParams->kSize))
            {
                throw Exception(1,"[localavgpool] kSize can't convert to int", __FILE__, __LINE__, __FUNCTION__);
            }
        }
        else if(key == "kSizeX")
        {
            if(!ExString::strToInt(value, localAvgPoolParams->kSizeX))
            {
                throw Exception(1,"[localavgpool] kSizeX can't convert to int", __FILE__, __LINE__, __FUNCTION__);
            }
        }
        else if(key == "kSizeY")
        {
            if(!ExString::strToInt(value, localAvgPoolParams->kSizeY))
            {
                throw Exception(1,"[localavgpool] kSizeY can't convert to int", __FILE__, __LINE__, __FUNCTION__);
            }
        }
        else if(key == "stride")
        {
            if(!ExString::strToInt(value, localAvgPoolParams->stride))
            {
                throw Exception(1,"[localavgpool] stride can't convert to int", __FILE__, __LINE__, __FUNCTION__);
            }
        }
        else if(key == "strideX")
        {
            if(!ExString::strToInt(value, localAvgPoolParams->strideX))
            {
                throw Exception(1,"[localavgpool] strideX can't convert to int", __FILE__, __LINE__, __FUNCTION__);
            }
        }
        else if(key == "strideY")
        {
            if(!ExString::strToInt(value, localAvgPoolParams->strideY))
            {
                throw Exception(1,"[localavgpool] strideY can't convert to int", __FILE__, __LINE__, __FUNCTION__);
            }
        }
        else if(key == "padding")
        {
            if(!ExString::strToInt(value, localAvgPoolParams->padding))
            {
                throw Exception(1,"[localavgpool] padding can't convert to int", __FILE__, __LINE__, __FUNCTION__);
            }
        }
        else if(key == "paddingX")
        {
            if(!ExString::strToInt(value, localAvgPoolParams->paddingX))
            {
                throw Exception(1,"[localavgpool] paddingX can't convert to int", __FILE__, __LINE__, __FUNCTION__);
            }
        }
        else if(key == "paddingY")
        {
            if(!ExString::strToInt(value, localAvgPoolParams->paddingY))
            {
                throw Exception(1,"[localavgpool] paddingY can't convert to int", __FILE__, __LINE__, __FUNCTION__);
            }
        }
        else if(key == "ceilMode")
        {
            if(!ExString::strToInt(value, localAvgPoolParams->ceilMode))
            {
                throw Exception(1,"[localavgpool] ceilMode can't convert to int", __FILE__, __LINE__, __FUNCTION__);
            }
        }
        else
        {
            throw Exception(1, key + " is not supported in [localavgpool]", __FILE__, __LINE__, __FUNCTION__);
        }
    }

    if(localAvgPoolParams->strideX < 1 || localAvgPoolParams->strideY < 1)
    {
        if(localAvgPoolParams->strideX < 1 )
        {
            localAvgPoolParams->strideX = localAvgPoolParams->stride;
        }

        if(localAvgPoolParams->strideY < 1 )
        {
            localAvgPoolParams->strideY = localAvgPoolParams->stride;
        }
    }

    if(localAvgPoolParams->paddingX < 1 || localAvgPoolParams->paddingY < 1)
    {
        if(localAvgPoolParams->paddingX < 1 )
        {
            localAvgPoolParams->paddingX = localAvgPoolParams->padding;
        }

        if(localAvgPoolParams->paddingY < 1 )
        {
            localAvgPoolParams->paddingY = localAvgPoolParams->padding;
        }
    }

    if(localAvgPoolParams->kSizeX < 1 || localAvgPoolParams->kSizeY < 1)
    {
        if(localAvgPoolParams->kSizeX < 1 )
        {
            localAvgPoolParams->kSizeX = localAvgPoolParams->kSize;
        }

        if(localAvgPoolParams->kSizeY < 1 )
        {
            localAvgPoolParams->kSizeY = localAvgPoolParams->kSize;
        }
    }
}

void Parser::parseGlobalAvgPoolParams(GlobalAvgPoolParams *globalAvgPoolParams, YAML::const_iterator &iter)
{
    (void)iter;
    (void)globalAvgPoolParams;
}

void Parser::parseConvParams(ConvParams *convParams, YAML::const_iterator &iter)
{
    for (YAML::const_iterator it = iter->second.begin(); it != iter->second.end(); ++it)
    {
        std::string key     =   it->first.as<std::string>();
        std::string value   =   it->second.as<std::string>();

        if(key == "batchNorm")
        {
            if(!ExString::strToInt(value, convParams->batchNorm))
            {
                throw Exception(1,"[conv] batchNorm can't convert to int", __FILE__, __LINE__, __FUNCTION__);
            }
        }
        else if(key == "filters")
        {
            if(!ExString::strToInt(value, convParams->filters))
            {
                throw Exception(1,"[conv] filters can't convert to int", __FILE__, __LINE__, __FUNCTION__);
            }
        }
        else if(key == "groups")
        {
            if(!ExString::strToInt(value, convParams->groups))
            {
                throw Exception(1,"[conv] groups can't convert to int", __FILE__, __LINE__, __FUNCTION__);
            }
        }
        else if(key == "kSize")
        {
            if(!ExString::strToInt(value, convParams->kSize))
            {
                throw Exception(1,"[conv] kSize can't convert to int", __FILE__, __LINE__, __FUNCTION__);
            }
        }
        else if(key == "kSizeX")
        {
            if(!ExString::strToInt(value, convParams->kSizeX))
            {
                throw Exception(1,"[conv] kSizeX can't convert to int", __FILE__, __LINE__, __FUNCTION__);
            }
        }
        else if(key == "kSizeY")
        {
            if(!ExString::strToInt(value, convParams->kSizeY))
            {
                throw Exception(1,"[conv] kSizeY can't convert to int", __FILE__, __LINE__, __FUNCTION__);
            }
        }
        else if(key == "stride")
        {
            if(!ExString::strToInt(value, convParams->stride))
            {
                throw Exception(1,"[conv] stride can't convert to int", __FILE__, __LINE__, __FUNCTION__);
            }
        }
        else if(key == "strideX")
        {
            if(!ExString::strToInt(value, convParams->strideX))
            {
                throw Exception(1,"[conv] strideX can't convert to int", __FILE__, __LINE__, __FUNCTION__);
            }
        }
        else if(key == "strideY")
        {
            if(!ExString::strToInt(value, convParams->strideY))
            {
                throw Exception(1,"[conv] strideY can't convert to int", __FILE__, __LINE__, __FUNCTION__);
            }
        }
        else if(key == "dilation")
        {
            if(!ExString::strToInt(value, convParams->dilation))
            {
                throw Exception(1,"[conv] dilation can't convert to int", __FILE__, __LINE__, __FUNCTION__);
            }
        }
        else if(key == "dilationX")
        {
            if(!ExString::strToInt(value, convParams->dilationX))
            {
                throw Exception(1,"[conv] dilation can't convert to int", __FILE__, __LINE__, __FUNCTION__);
            }
        }
        else if(key == "dilationY")
        {
            if(!ExString::strToInt(value, convParams->dilationY))
            {
                throw Exception(1,"[conv] dilation can't convert to int", __FILE__, __LINE__, __FUNCTION__);
            }
        }
        else if(key == "antialiasing")
        {
            if(!ExString::strToInt(value, convParams->antialiasing))
            {
                throw Exception(1,"[conv] antialiasing can't convert to int", __FILE__, __LINE__, __FUNCTION__);
            }
        }
        else if(key == "padding")
        {
            if(!ExString::strToInt(value, convParams->padding))
            {
                throw Exception(1,"[conv] padding can't convert to int", __FILE__, __LINE__, __FUNCTION__);
            }
        }
        else if(key == "paddingX")
        {
            if(!ExString::strToInt(value, convParams->paddingX))
            {
                throw Exception(1,"[conv] paddingX can't convert to int", __FILE__, __LINE__, __FUNCTION__);
            }
        }
        else if(key == "paddingY")
        {
            if(!ExString::strToInt(value, convParams->paddingY))
            {
                throw Exception(1,"[conv] paddingY can't convert to int", __FILE__, __LINE__, __FUNCTION__);
            }
        }
        else if(key == "useBias")
        {
            if(!ExString::strToInt(value, convParams->useBias))
            {
                throw Exception(1,"[conv] useBias can't convert to int", __FILE__, __LINE__, __FUNCTION__);
            }
        }
        else if(key == "activation")
        {
            std::vector<std::string> splits;
            ExString::split(splits, value, ",");
            convParams->activation = Activations::getActivation(splits[0]);

            if(splits.size()>1)
            {
                for (size_t i = 1; i < splits.size(); ++i)
                {
                    float tmp = 0.f;
                    ExString::strToFloat(splits[i], tmp);
                    convParams->actParams.push_back(tmp);
                }
            }
        }
        else
        {
            throw Exception(1, key + " is not supported in [conv]", __FILE__, __LINE__, __FUNCTION__);
        }
    }

    if(convParams->strideX < 0 || convParams->strideY < 0)
    {
        if(convParams->strideX < 0 )
        {
            convParams->strideX = convParams->stride;
        }

        if(convParams->strideY < 0 )
        {
            convParams->strideY = convParams->stride;
        }
    }

    if(convParams->kSizeX < 0 || convParams->kSizeY < 0)
    {
        if(convParams->kSizeX < 0 )
        {
            convParams->kSizeX = convParams->kSize;
        }

        if(convParams->kSizeY < 0 )
        {
            convParams->kSizeY = convParams->kSize;
        }
    }

    if(convParams->paddingX < 0 || convParams->paddingY < 0)
    {
        if(convParams->paddingX < 0 )
        {
            convParams->paddingX = convParams->padding;
        }

        if(convParams->paddingY < 0 )
        {
            convParams->paddingY = convParams->padding;
        }
    }

    if(convParams->dilationX < 0 || convParams->dilationY < 0)
    {
        if(convParams->dilationX < 0 )
        {
            convParams->dilationX = convParams->dilation;
        }

        if(convParams->dilationY < 0 )
        {
            convParams->dilationY = convParams->dilation;
        }
    }

    if(convParams->kSizeX == 1)
    {
        if(convParams->dilationX > 1)
        {
            convParams->dilationX = 1;
        }
    }

    if(convParams->kSizeY == 1)
    {
        if(convParams->dilationY > 1)
        {
            convParams->dilationY = 1;
        }
    }

}

void Parser::parseDeConvParams(DeConvParams *deconvParams, YAML::const_iterator &iter)
{
    for (YAML::const_iterator it = iter->second.begin(); it != iter->second.end(); ++it)
    {
        std::string key     =   it->first.as<std::string>();
        std::string value   =   it->second.as<std::string>();

        if(key == "filters")
        {
            if(!ExString::strToInt(value, deconvParams->filters))
            {
                throw Exception(1,"[deconv] filters can't convert to int", __FILE__, __LINE__, __FUNCTION__);
            }
        }
        else if(key == "kSize")
        {
            if(!ExString::strToInt(value, deconvParams->kSize))
            {
                throw Exception(1,"[deconv] kSize can't convert to int", __FILE__, __LINE__, __FUNCTION__);
            }
        }
        else if(key == "kSizeX")
        {
            if(!ExString::strToInt(value, deconvParams->kSizeX))
            {
                throw Exception(1,"[deconv] kSizeX can't convert to int", __FILE__, __LINE__, __FUNCTION__);
            }
        }
        else if(key == "kSizeY")
        {
            if(!ExString::strToInt(value, deconvParams->kSizeY))
            {
                throw Exception(1,"[deconv] kSizeY can't convert to int", __FILE__, __LINE__, __FUNCTION__);
            }
        }
        else if(key == "stride")
        {
            if(!ExString::strToInt(value, deconvParams->stride))
            {
                throw Exception(1,"[deconv] stride can't convert to int", __FILE__, __LINE__, __FUNCTION__);
            }
        }
        else if(key == "strideX")
        {
            if(!ExString::strToInt(value, deconvParams->strideX))
            {
                throw Exception(1,"[deconv] strideX can't convert to int", __FILE__, __LINE__, __FUNCTION__);
            }
        }
        else if(key == "strideY")
        {
            if(!ExString::strToInt(value, deconvParams->strideY))
            {
                throw Exception(1,"[deconv] strideY can't convert to int", __FILE__, __LINE__, __FUNCTION__);
            }
        }
        else if(key == "padding")
        {
            if(!ExString::strToInt(value, deconvParams->padding))
            {
                throw Exception(1,"[deconv] padding can't convert to int", __FILE__, __LINE__, __FUNCTION__);
            }
        }
        else if(key == "paddingX")
        {
            if(!ExString::strToInt(value, deconvParams->paddingX))
            {
                throw Exception(1,"[deconv] paddingX can't convert to int", __FILE__, __LINE__, __FUNCTION__);
            }
        }
        else if(key == "paddingY")
        {
            if(!ExString::strToInt(value, deconvParams->paddingY))
            {
                throw Exception(1,"[deconv] paddingY can't convert to int", __FILE__, __LINE__, __FUNCTION__);
            }
        }
        else if(key == "useBias")
        {
            if(!ExString::strToInt(value, deconvParams->useBias))
            {
                throw Exception(1,"[deconv] useBias can't convert to int", __FILE__, __LINE__, __FUNCTION__);
            }
        }
        else if(key == "groups")
        {
            if(!ExString::strToInt(value, deconvParams->groups))
            {
                throw Exception(1,"[deconv] groups can't convert to int", __FILE__, __LINE__, __FUNCTION__);
            }
        }
        else if(key == "activation")
        {
            std::vector<std::string> splits;
            ExString::split(splits, value, ",");
            deconvParams->activation = Activations::getActivation(splits[0]);

            if(splits.size()>1)
            {
                for (size_t i = 1; i < splits.size(); ++i)
                {
                    float tmp = 0.f;
                    ExString::strToFloat(splits[i], tmp);
                    deconvParams->actParams.push_back(tmp);
                }
            }
        }
        else
        {
            throw Exception(1, key + " is not supported in [deconv]", __FILE__, __LINE__, __FUNCTION__);
        }
    }

    if(deconvParams->strideX < 0 || deconvParams->strideY < 0)
    {
        if(deconvParams->strideX < 0 )
        {
            deconvParams->strideX = deconvParams->stride;
        }

        if(deconvParams->strideY < 0 )
        {
            deconvParams->strideY = deconvParams->stride;
        }
    }

    if(deconvParams->kSizeX < 0 || deconvParams->kSizeY < 0)
    {
        if(deconvParams->kSizeX < 0 )
        {
            deconvParams->kSizeX = deconvParams->kSize;
        }

        if(deconvParams->kSizeY < 0 )
        {
            deconvParams->kSizeY = deconvParams->kSize;
        }
    }

    if(deconvParams->paddingX < 0 || deconvParams->paddingY < 0)
    {
        if(deconvParams->paddingX < 0 )
        {
            deconvParams->paddingX = deconvParams->padding;
        }

        if(deconvParams->paddingY < 0 )
        {
            deconvParams->paddingY = deconvParams->padding;
        }
    }

}

void Parser::parseConnectParams(ConnectParams *connectParams, YAML::const_iterator &iter)
{
    for (YAML::const_iterator it = iter->second.begin(); it != iter->second.end(); ++it)
    {
        std::string key     =   it->first.as<std::string>();
        std::string value   =   it->second.as<std::string>();

        if(key == "output")
        {
            if(!ExString::strToInt(value, connectParams->output))
            {
                throw Exception(1,"[connect] output can't convert to int", __FILE__, __LINE__, __FUNCTION__);
            }
        }
        else if(key == "batchNorm")
        {
            if(!ExString::strToInt(value, connectParams->batchNorm))
            {
                throw Exception(1,"[connect] batchNorm can't convert to int", __FILE__, __LINE__, __FUNCTION__);
            }
        }
        else if(key == "activation")
        {
            std::vector<std::string> splits;
            ExString::split(splits, value, ",");
            connectParams->activation = Activations::getActivation(splits[0]);

            if(splits.size()>1)
            {
                for (size_t i = 1; i < splits.size(); ++i)
                {
                    float tmp = 0.f;
                    ExString::strToFloat(splits[i], tmp);
                    connectParams->actParams.push_back(tmp);
                }
            }
        }
        else
        {
            throw Exception(1, key + " is not supported in [connect]", __FILE__, __LINE__, __FUNCTION__);
        }
    }
}

void Parser::parseBatchNormParams(BatchNormParams *batchNormParams, YAML::const_iterator &iter)
{
    for (YAML::const_iterator it = iter->second.begin(); it != iter->second.end(); ++it)
    {
        std::string key     =   it->first.as<std::string>();
        std::string value   =   it->second.as<std::string>();

        if(key == "activation")
        {
            std::vector<std::string> splits;
            ExString::split(splits, value, ",");
            batchNormParams->activation = Activations::getActivation(splits[0]);

            if(splits.size()>1)
            {
                for (size_t i = 1; i < splits.size(); ++i)
                {
                    float tmp = 0.f;
                    ExString::strToFloat(splits[i], tmp);
                    batchNormParams->actParams.push_back(tmp);
                }
            }
        }
        else
        {
            throw Exception(1, key + " is not supported in [batchnorm]", __FILE__, __LINE__, __FUNCTION__);
        }
    }
}

void Parser::parseEmptyNormParams(EmptyParams *emptyParams, YAML::const_iterator &iter)
{
    (void)iter;
    (void)emptyParams;
}

void Parser::parsePaddingParams(PaddingParams *paddingParams, YAML::const_iterator &iter)
{
    for (YAML::const_iterator it = iter->second.begin(); it != iter->second.end(); ++it)
    {
        std::string key     =   it->first.as<std::string>();
        std::string value   =   it->second.as<std::string>();

        if(key == "top")
        {
            if(!ExString::strToInt(value, paddingParams->top))
            {
                throw Exception(1,"[padding] top can't convert to int", __FILE__, __LINE__, __FUNCTION__);
            }
        }
        else if(key == "down")
        {
            if(!ExString::strToInt(value, paddingParams->down))
            {
                throw Exception(1,"[padding] down can't convert to int", __FILE__, __LINE__, __FUNCTION__);
            }
        }
        else if(key == "left")
        {
            if(!ExString::strToInt(value, paddingParams->left))
            {
                throw Exception(1,"[padding] left can't convert to int", __FILE__, __LINE__, __FUNCTION__);
            }
        }
        else if(key == "right")
        {
            if(!ExString::strToInt(value, paddingParams->right))
            {
                throw Exception(1,"[padding] right can't convert to int", __FILE__, __LINE__, __FUNCTION__);
            }
        }
        else if(key == "paddingVal")
        {
            if(!ExString::strToFloat(value, paddingParams->paddingVal))
            {
                throw Exception(1,"[padding] paddingVal can't convert to int", __FILE__, __LINE__, __FUNCTION__);
            }
        }
        else
        {
            throw Exception(1, key + " is not supported in [padding]", __FILE__, __LINE__, __FUNCTION__);
        }
    }
}

void Parser::parseResBlockParams(ResBlockParams *resBlockParams, YAML::const_iterator &iter)
{

    for (YAML::const_iterator it = iter->second.begin(); it != iter->second.end(); ++it)
    {
        std::string key     =   it->first.as<std::string>();

        if(iter->second.Type() == YAML::NodeType::Map)
        {
            if(key == "size" || key == "activation")
            {

            }
            else if(key == "maxpool")
            {
                MaxPoolParams *maxPoolParams = new MaxPoolParams(false);
                parseMaxPoolParams(maxPoolParams, it);
                resBlockParams->baseParams.push_back(maxPoolParams);
            }
            else if(key == "localavgpool")
            {
                LocalAvgPoolParams *localAvgPoolParams = new LocalAvgPoolParams(false);
                parseLocalAvgPoolParams(localAvgPoolParams, it);
                resBlockParams->baseParams.push_back(localAvgPoolParams);
            }
            else if(key == "conv")
            {
                ConvParams *convParams = new ConvParams(false);
                parseConvParams(convParams, it);
                resBlockParams->baseParams.push_back(convParams);
            }
            else if(key == "connect")
            {
                ConnectParams *connectParams = new ConnectParams(false);
                parseConnectParams(connectParams, it);
                resBlockParams->baseParams.push_back(connectParams);
            }
            else if(key == "batchnorm")
            {
                BatchNormParams *batchNormParams = new BatchNormParams(false);
                parseBatchNormParams(batchNormParams, it);
                resBlockParams->baseParams.push_back(batchNormParams);
            }
            else if(key == "padding")
            {
                PaddingParams *paddingParams = new PaddingParams(false);
                parsePaddingParams(paddingParams, it);
                resBlockParams->baseParams.push_back(paddingParams);
            }
            else
            {
                throw Exception(1, key +  " is not supported by [resblock]", __FILE__, __LINE__, __FUNCTION__);
            }
        }
    }
}

void Parser::parseRes2BlockParams(Res2BlockParams *res2BlockParams, YAML::const_iterator &iter)
{
    for (YAML::const_iterator it = iter->second.begin(); it != iter->second.end(); ++it)
    {
        std::string key     =   it->first.as<std::string>();

        if(iter->second.Type() == YAML::NodeType::Map)
        {
            if(key == "base")
            {
                for (YAML::const_iterator it2 = it->second.begin(); it2 != it->second.end(); ++it2)
                {
                    std::string key2     =   it2->first.as<std::string>();

                    if(key2 == "maxpool")
                    {
                        MaxPoolParams *maxPoolParams = new MaxPoolParams(false);
                        parseMaxPoolParams(maxPoolParams, it2);
                        res2BlockParams->baseParams.push_back(maxPoolParams);
                    }
                    else if(key2 == "localavgpool")
                    {
                        LocalAvgPoolParams *localAvgPoolParams = new LocalAvgPoolParams(false);
                        parseLocalAvgPoolParams(localAvgPoolParams, it2);
                        res2BlockParams->baseParams.push_back(localAvgPoolParams);
                    }
                    else if(key2 == "conv")
                    {
                        ConvParams *convParams = new ConvParams(false);
                        parseConvParams(convParams, it2);
                        res2BlockParams->baseParams.push_back(convParams);
                    }
                    else if(key2 == "connect")
                    {
                        ConnectParams *connectParams = new ConnectParams(false);
                        parseConnectParams(connectParams, it2);
                        res2BlockParams->baseParams.push_back(connectParams);
                    }
                    else if(key2 == "batchnorm")
                    {
                        BatchNormParams *batchNormParams = new BatchNormParams(false);
                        parseBatchNormParams(batchNormParams, it2);
                        res2BlockParams->baseParams.push_back(batchNormParams);
                    }
                    else if(key2 == "padding")
                    {
                        PaddingParams *paddingParams = new PaddingParams(false);
                        parsePaddingParams(paddingParams, it2);
                        res2BlockParams->baseParams.push_back(paddingParams);
                    }
                    else
                    {
                        throw Exception(1, key2 +  " is not supported by [res2block]", __FILE__, __LINE__, __FUNCTION__);
                    }
                }
            }
            else if(key == "branch")
            {
                for (YAML::const_iterator it2 = it->second.begin(); it2 != it->second.end(); ++it2)
                {
                    std::string key2     =   it2->first.as<std::string>();

                    if(key2 == "maxpool")
                    {
                        MaxPoolParams *maxPoolParams = new MaxPoolParams(false);
                        parseMaxPoolParams(maxPoolParams, it2);
                        res2BlockParams->branchParams.push_back(maxPoolParams);
                    }
                    else if(key2 == "localavgpool")
                    {
                        LocalAvgPoolParams *localAvgPoolParams = new LocalAvgPoolParams(false);
                        parseLocalAvgPoolParams(localAvgPoolParams, it2);
                        res2BlockParams->branchParams.push_back(localAvgPoolParams);
                    }
                    else if(key2 == "conv")
                    {
                        ConvParams *convParams = new ConvParams(false);
                        parseConvParams(convParams, it2);
                        res2BlockParams->branchParams.push_back(convParams);
                    }
                    else if(key2 == "connect")
                    {
                        ConnectParams *connectParams = new ConnectParams(false);
                        parseConnectParams(connectParams, it2);
                        res2BlockParams->branchParams.push_back(connectParams);
                    }
                    else if(key2 == "batchnorm")
                    {
                        BatchNormParams *batchNormParams = new BatchNormParams(false);
                        parseBatchNormParams(batchNormParams, it2);
                        res2BlockParams->branchParams.push_back(batchNormParams);
                    }
                    else if(key2 == "padding")
                    {
                        PaddingParams *paddingParams = new PaddingParams(false);
                        parsePaddingParams(paddingParams, it2);
                        res2BlockParams->branchParams.push_back(paddingParams);
                    }
                    else
                    {
                        throw Exception(1, key2 +  " is not supported by [res2block]", __FILE__, __LINE__, __FUNCTION__);
                    }
                }
            }
        }
    }
}

void Parser::parseConcatBlockParams(ConcatBlockParams *concatBlockParams, YAML::const_iterator &iter)
{

    for (YAML::const_iterator it = iter->second.begin(); it != iter->second.end(); ++it)
    {
        std::string key     =   it->first.as<std::string>();

        if(iter->second.Type() == YAML::NodeType::Map)
        {

            std::vector<BaseParams*> tmpParams;

            if(key == "branch")
            {
                for (YAML::const_iterator it2 = it->second.begin(); it2 != it->second.end(); ++it2)
                {
                    std::string key2     =   it2->first.as<std::string>();

                    if(key2 == "maxpool")
                    {
                        MaxPoolParams *maxPoolParams = new MaxPoolParams(false);
                        parseMaxPoolParams(maxPoolParams, it2);
                        tmpParams.push_back(maxPoolParams);
                    }
                    else if(key2 == "localavgpool")
                    {
                        LocalAvgPoolParams *localAvgPoolParams = new LocalAvgPoolParams(false);
                        parseLocalAvgPoolParams(localAvgPoolParams, it2);
                        tmpParams.push_back(localAvgPoolParams);
                    }
                    else if(key2 == "conv")
                    {
                        ConvParams *convParams = new ConvParams(false);
                        parseConvParams(convParams, it2);
                        tmpParams.push_back(convParams);
                    }
                    else if(key2 == "connect")
                    {
                        ConnectParams *connectParams = new ConnectParams(false);
                        parseConnectParams(connectParams, it2);
                        tmpParams.push_back(connectParams);
                    }
                    else if(key2 == "batchnorm")
                    {
                        BatchNormParams *batchNormParams = new BatchNormParams(false);
                        parseBatchNormParams(batchNormParams, it2);
                        tmpParams.push_back(batchNormParams);
                    }
                    else if(key2 == "empty")
                    {
                        EmptyParams *emptyParams = new EmptyParams(false);
                        parseEmptyNormParams(emptyParams, it2);
                        tmpParams.push_back(emptyParams);
                    }
                    else if(key2 == "concatblock")
                    {
                        ConcatBlockParams *concatBlockParams1 = new ConcatBlockParams(false);
                        parseConcatBlockParams(concatBlockParams1, it2);
                        tmpParams.push_back(concatBlockParams1);
                    }
                    else if(key2 == "addblock")
                    {
                        AddBlockParams *addBlockParams = new AddBlockParams(false);
                        parseAddBlockParams(addBlockParams, it2);
                        tmpParams.push_back(addBlockParams);
                    }
                    else if(key2 == "padding")
                    {
                        PaddingParams *paddingParams = new PaddingParams(false);
                        parsePaddingParams(paddingParams, it2);
                        tmpParams.push_back(paddingParams);
                    }
                    else
                    {
                        throw Exception(1, key2 +  " is not supported by [concatblock]", __FILE__, __LINE__, __FUNCTION__);
                    }
                }
            }

            concatBlockParams->branchParams.push_back(tmpParams);
        }
    }
}

void Parser::parseAddBlockParams(AddBlockParams *addBlockParams, YAML::const_iterator &iter)
{

    for (YAML::const_iterator it = iter->second.begin(); it != iter->second.end(); ++it)
    {
        std::string key     =   it->first.as<std::string>();

        if(iter->second.Type() == YAML::NodeType::Map)
        {

            std::vector<BaseParams*> tmpParams;

            if(key == "branch")
            {
                for (YAML::const_iterator it2 = it->second.begin(); it2 != it->second.end(); ++it2)
                {
                    std::string key2     =   it2->first.as<std::string>();

                    if(key2 == "maxpool")
                    {
                        MaxPoolParams *maxPoolParams = new MaxPoolParams(false);
                        parseMaxPoolParams(maxPoolParams, it2);
                        tmpParams.push_back(maxPoolParams);
                    }
                    else if(key2 == "localavgpool")
                    {
                        LocalAvgPoolParams *localAvgPoolParams = new LocalAvgPoolParams(false);
                        parseLocalAvgPoolParams(localAvgPoolParams, it2);
                        tmpParams.push_back(localAvgPoolParams);
                    }
                    else if(key2 == "conv")
                    {
                        ConvParams *convParams = new ConvParams(false);
                        parseConvParams(convParams, it2);
                        tmpParams.push_back(convParams);
                    }
                    else if(key2 == "connect")
                    {
                        ConnectParams *connectParams = new ConnectParams(false);
                        parseConnectParams(connectParams, it2);
                        tmpParams.push_back(connectParams);
                    }
                    else if(key2 == "batchnorm")
                    {
                        BatchNormParams *batchNormParams = new BatchNormParams(false);
                        parseBatchNormParams(batchNormParams, it2);
                        tmpParams.push_back(batchNormParams);
                    }
                    else if(key2 == "empty")
                    {
                        EmptyParams *emptyParams = new EmptyParams(false);
                        parseEmptyNormParams(emptyParams, it2);
                        tmpParams.push_back(emptyParams);
                    }
                    else if(key2 == "concatblock")
                    {
                        ConcatBlockParams *concatBlockParams1 = new ConcatBlockParams(false);
                        parseConcatBlockParams(concatBlockParams1, it2);
                        tmpParams.push_back(concatBlockParams1);
                    }
                    else if(key2 == "addblock")
                    {
                        AddBlockParams *addBlockParams = new AddBlockParams(false);
                        parseAddBlockParams(addBlockParams, it2);
                        tmpParams.push_back(addBlockParams);
                    }
                    else if(key2 == "padding")
                    {
                        PaddingParams *paddingParams = new PaddingParams(false);
                        parsePaddingParams(paddingParams, it2);
                        tmpParams.push_back(paddingParams);
                    }
                    else
                    {
                        throw Exception(1, key2 +  " is not supported by [addblock]", __FILE__, __LINE__, __FUNCTION__);
                    }
                }

                addBlockParams->branchParams.push_back(tmpParams);
            }

        }
    }
}

void Parser::parseRouteParams(RouteParams *routeParams, YAML::const_iterator &iter)
{
    for (YAML::const_iterator it = iter->second.begin(); it != iter->second.end(); ++it)
    {
        std::string key     =   it->first.as<std::string>();
        std::string value   =   it->second.as<std::string>();

        if(key == "layers")
        {
            std::vector<std::string> layerIndexes;
            ExString::split(layerIndexes, value, ",");

            for (size_t i = 0; i < layerIndexes.size(); ++i)
            {
                int index   =  0;

                if(!ExString::strToInt(layerIndexes[i], index))
                {
                    throw Exception(1,"[route] kSize can't convert to int", __FILE__, __LINE__, __FUNCTION__);
                }

                if(index < 0)
                {
                    index   = index + routeParams->index - 1;
                }

                routeParams->layerIndexes.push_back(index);
            }
        }
        else if(key == "groups")
        {
            if(!ExString::strToInt(value, routeParams->groups))
            {
                throw Exception(1,"[route] output can't convert to int", __FILE__, __LINE__, __FUNCTION__);
            }
        }
        else if(key == "groupsId")
        {
            if(!ExString::strToInt(value, routeParams->groupsId))
            {
                throw Exception(1,"[route] output can't convert to int", __FILE__, __LINE__, __FUNCTION__);
            }
        }
        else if(key == "addModel")
        {
            if(!ExString::strToInt(value, routeParams->addModel))
            {
                throw Exception(1,"[route] output can't convert to int", __FILE__, __LINE__, __FUNCTION__);
            }
        }
        else
        {
            throw Exception(1, key + " is not supported in [route]", __FILE__, __LINE__, __FUNCTION__);
        }
    }
}

void Parser::parseSoftMaxParams(SoftMaxParams *softmaxParams, YAML::const_iterator &iter)
{
    for (YAML::const_iterator it = iter->second.begin(); it != iter->second.end(); ++it)
    {
        std::string key     =   it->first.as<std::string>();
        std::string value   =   it->second.as<std::string>();

        if(key == "groups")
        {
            if(!ExString::strToInt(value, softmaxParams->groups))
            {
                throw Exception(1,"[softmax] output can't convert to int", __FILE__, __LINE__, __FUNCTION__);
            }
        }
        else if(key == "temperature")
        {
            if(!ExString::strToFloat(value, softmaxParams->temperature))
            {
                throw Exception(1,"[softmax] output can't convert to int", __FILE__, __LINE__, __FUNCTION__);
            }
        }
        else
        {
            throw Exception(1, key + " is not supported in [softmax]", __FILE__, __LINE__, __FUNCTION__);
        }
    }
}

void Parser::parseUpSampleParams(UpSampleParams *upSampleParams, YAML::const_iterator &iter)
{
    for (YAML::const_iterator it = iter->second.begin(); it != iter->second.end(); ++it)
    {
        std::string key     =   it->first.as<std::string>();
        std::string value   =   it->second.as<std::string>();

        if(key == "stride")
        {
            if(!ExString::strToInt(value, upSampleParams->stride))
            {
                throw Exception(1,"[unsample] output can't convert to int", __FILE__, __LINE__, __FUNCTION__);
            }
        }
        else if(key == "scale")
        {
            if(!ExString::strToFloat(value, upSampleParams->scale))
            {
                throw Exception(1,"[unsample] output can't convert to int", __FILE__, __LINE__, __FUNCTION__);
            }
        }
        else
        {
            throw Exception(1, key + " is not supported in [unsample]", __FILE__, __LINE__, __FUNCTION__);
        }
    }
}

void Parser::parseYolov3Params(Yolov3Params *yolov3Params, YAML::const_iterator &iter)
{
    for (YAML::const_iterator it = iter->second.begin(); it != iter->second.end(); ++it)
    {
        std::string key     =   it->first.as<std::string>();
        std::string value   =   it->second.as<std::string>();

        if(key == "anchors")
        {
            std::vector<std::string> tmpAnchors;

            ExString::split(tmpAnchors, value, ",");

            if(tmpAnchors.size()!=6)
            {
                throw Exception(1,"[yolov3] anchor num should be 6", __FILE__, __LINE__, __FUNCTION__);
            }

            for (size_t i = 0; i < tmpAnchors.size(); ++i)
            {
                float tmp = 0;
                ExString::trim(tmpAnchors[i]);
                if(!ExString::strToFloat(tmpAnchors[i],tmp))
                {
                    throw Exception(1,"[yolov3] anchors can't convert to float", __FILE__, __LINE__, __FUNCTION__);
                }

                yolov3Params->anchors.push_back(tmp);
            }

        }
        else if(key == "classNum")
        {
            if(!ExString::strToInt(value, yolov3Params->classNum))
            {
                throw Exception(1,"[yolov3] output can't convert to int", __FILE__, __LINE__, __FUNCTION__);
            }
        }
        else
        {
            throw Exception(1, key + " is not supported in [unsample]", __FILE__, __LINE__, __FUNCTION__);
        }
    }
}

void Parser::parseYolov3OutParams(Yolov3OutParams *yolov3OutParams, YAML::const_iterator &iter)
{
    for (YAML::const_iterator it = iter->second.begin(); it != iter->second.end(); ++it)
    {
        std::string key     =   it->first.as<std::string>();
        std::string value   =   it->second.as<std::string>();

        if(key == "layers")
        {
            std::vector<std::string> layerIndexes;
            ExString::split(layerIndexes, value, ",");

            for (size_t i = 0; i < layerIndexes.size(); ++i)
            {
                int index   =  0;

                if(!ExString::strToInt(layerIndexes[i], index))
                {
                    throw Exception(1,"[yolov3out] kSize can't convert to int", __FILE__, __LINE__, __FUNCTION__);
                }

                if(index < 0)
                {
                    index   = index + yolov3OutParams->index - 1;
                }

                yolov3OutParams->layerIndexes.push_back(index);
            }
        }
        else if(key == "confThresh")
        {
            if(!ExString::strToFloat(value, yolov3OutParams->confThresh))
            {
                throw Exception(1,"[yolov3out] output can't convert to int", __FILE__, __LINE__, __FUNCTION__);
            }
        }
        else if(key == "nmsThresh")
        {
            if(!ExString::strToFloat(value, yolov3OutParams->nmsThresh))
            {
                throw Exception(1,"[yolov3out] output can't convert to int", __FILE__, __LINE__, __FUNCTION__);
            }
        }
        else if(key == "useSoftNms")
        {
            if(!ExString::strToInt(value, yolov3OutParams->useSoftNms))
            {
                throw Exception(1,"[yolov3out] output can't convert to int", __FILE__, __LINE__, __FUNCTION__);
            }
        }
        else if(key == "yoloType")
        {
            yolov3OutParams->getYoloTypeFromStr(value);
        }
        else
        {
            throw Exception(1, key + " is not supported in [yolov3out]", __FILE__, __LINE__, __FUNCTION__);
        }
    }
}

ConcatBlockParams::~ConcatBlockParams()
{
    for (size_t i = 0; i < branchParams.size(); ++i)
    {
        for (size_t j = 0; j < branchParams[i].size(); ++j)
        {
            if(branchParams[i][j]!=nullptr)
            {

                if(branchParams[i][j]->type == LayerType::CONFIG)
                {
                    delete reinterpret_cast<NetConfigParams*>(branchParams[i][j]);
                }
                else if(branchParams[i][j]->type == LayerType::CONVOLUTIONAL)
                {
                    delete reinterpret_cast<ConvParams*>(branchParams[i][j]);
                }
                else if(branchParams[i][j]->type == LayerType::MAXPOOL)
                {
                    delete reinterpret_cast<MaxPoolParams*>(branchParams[i][j]);
                }
                else if(branchParams[i][j]->type == LayerType::LOCAL_AVGPOOL)
                {
                    delete reinterpret_cast<LocalAvgPoolParams*>(branchParams[i][j]);
                }
                else if(branchParams[i][j]->type == LayerType::CONNECTED)
                {
                    delete reinterpret_cast<ConnectParams*>(branchParams[i][j]);
                }
                else if(branchParams[i][j]->type == LayerType::BATCHNORM)
                {
                    delete reinterpret_cast<BatchNormParams*>(branchParams[i][j]);
                }
                else if(branchParams[i][j]->type == LayerType::EMPTY)
                {
                    delete reinterpret_cast<EmptyParams*>(branchParams[i][j]);
                }
                else if(branchParams[i][j]->type == LayerType::CONCAT_BLOCK)
                {
                    delete reinterpret_cast<ConcatBlockParams*>(branchParams[i][j]);
                }
                else if(branchParams[i][j]->type == LayerType::ADD_BLOCK)
                {
                    delete reinterpret_cast<AddBlockParams*>(branchParams[i][j]);
                }
                else if(branchParams[i][j]->type == LayerType::PADDING)
                {
                    delete reinterpret_cast<PaddingParams*>(branchParams[i][j]);
                }

                branchParams[i][j] = nullptr;
            }
        }

        if(i == (branchParams.size()-1))
        {
            branchParams.clear();
        }
    }
}

AddBlockParams::~AddBlockParams()
{
    for (size_t i = 0; i < branchParams.size(); ++i)
    {
        for (size_t j = 0; j < branchParams[i].size(); ++j)
        {
            if(branchParams[i][j]!=nullptr)
            {

                if(branchParams[i][j]->type == LayerType::CONFIG)
                {
                    delete reinterpret_cast<NetConfigParams*>(branchParams[i][j]);
                }
                else if(branchParams[i][j]->type == LayerType::CONVOLUTIONAL)
                {
                    delete reinterpret_cast<ConvParams*>(branchParams[i][j]);
                }
                else if(branchParams[i][j]->type == LayerType::MAXPOOL)
                {
                    delete reinterpret_cast<MaxPoolParams*>(branchParams[i][j]);
                }
                else if(branchParams[i][j]->type == LayerType::LOCAL_AVGPOOL)
                {
                    delete reinterpret_cast<LocalAvgPoolParams*>(branchParams[i][j]);
                }
                else if(branchParams[i][j]->type == LayerType::CONNECTED)
                {
                    delete reinterpret_cast<ConnectParams*>(branchParams[i][j]);
                }
                else if(branchParams[i][j]->type == LayerType::BATCHNORM)
                {
                    delete reinterpret_cast<BatchNormParams*>(branchParams[i][j]);
                }
                else if(branchParams[i][j]->type == LayerType::EMPTY)
                {
                    delete reinterpret_cast<EmptyParams*>(branchParams[i][j]);
                }
                else if(branchParams[i][j]->type == LayerType::CONCAT_BLOCK)
                {
                    delete reinterpret_cast<ConcatBlockParams*>(branchParams[i][j]);
                }
                else if(branchParams[i][j]->type == LayerType::ADD_BLOCK)
                {
                    delete reinterpret_cast<AddBlockParams*>(branchParams[i][j]);
                }
                else if(branchParams[i][j]->type == LayerType::PADDING)
                {
                    delete reinterpret_cast<PaddingParams*>(branchParams[i][j]);
                }

                branchParams[i][j] = nullptr;
            }
        }

        if(i == (branchParams.size()-1))
        {
            branchParams.clear();
        }
    }
}

Res2BlockParams::~Res2BlockParams()
{
    for (size_t i = 0; i < baseParams.size(); ++i)
    {
        if(baseParams[i]!=nullptr)
        {

            if(baseParams[i]->type == LayerType::CONFIG)
            {
                delete reinterpret_cast<NetConfigParams*>(baseParams[i]);
            }
            else if(baseParams[i]->type == LayerType::CONVOLUTIONAL)
            {
                delete reinterpret_cast<ConvParams*>(baseParams[i]);
            }
            else if(baseParams[i]->type == LayerType::MAXPOOL)
            {
                delete reinterpret_cast<MaxPoolParams*>(baseParams[i]);
            }
            else if(baseParams[i]->type == LayerType::LOCAL_AVGPOOL)
            {
                delete reinterpret_cast<LocalAvgPoolParams*>(baseParams[i]);
            }
            else if(baseParams[i]->type == LayerType::CONNECTED)
            {
                delete reinterpret_cast<ConnectParams*>(baseParams[i]);
            }
            else if(baseParams[i]->type == LayerType::BATCHNORM)
            {
                delete reinterpret_cast<BatchNormParams*>(baseParams[i]);
            }
            else if(baseParams[i]->type == LayerType::PADDING)
            {
                delete reinterpret_cast<PaddingParams*>(baseParams[i]);
            }

            baseParams[i] = nullptr;
        }

        if(i == (baseParams.size()-1))
        {
            baseParams.clear();
        }
    }

    for (size_t i = 0; i < branchParams.size(); ++i)
    {
        if(branchParams[i]!=nullptr)
        {

            if(branchParams[i]->type == LayerType::CONFIG)
            {
                delete reinterpret_cast<NetConfigParams*>(branchParams[i]);
            }
            else if(branchParams[i]->type == LayerType::CONVOLUTIONAL)
            {
                delete reinterpret_cast<ConvParams*>(branchParams[i]);
            }
            else if(branchParams[i]->type == LayerType::MAXPOOL)
            {
                delete reinterpret_cast<MaxPoolParams*>(branchParams[i]);
            }
            else if(branchParams[i]->type == LayerType::LOCAL_AVGPOOL)
            {
                delete reinterpret_cast<LocalAvgPoolParams*>(branchParams[i]);
            }
            else if(branchParams[i]->type == LayerType::CONNECTED)
            {
                delete reinterpret_cast<ConnectParams*>(branchParams[i]);
            }
            else if(branchParams[i]->type == LayerType::BATCHNORM)
            {
                delete reinterpret_cast<BatchNormParams*>(branchParams[i]);
            }

            branchParams[i] = nullptr;
        }

        if(i == (branchParams.size()-1))
        {
            branchParams.clear();
        }
    }
}

ResBlockParams::~ResBlockParams()
{
    for (size_t i = 0; i < baseParams.size(); ++i)
    {
        if(baseParams[i]!=nullptr)
        {

            if(baseParams[i]->type == LayerType::CONFIG)
            {
                delete reinterpret_cast<NetConfigParams*>(baseParams[i]);
            }
            else if(baseParams[i]->type == LayerType::CONVOLUTIONAL)
            {
                delete reinterpret_cast<ConvParams*>(baseParams[i]);
            }
            else if(baseParams[i]->type == LayerType::MAXPOOL)
            {
                delete reinterpret_cast<MaxPoolParams*>(baseParams[i]);
            }
            else if(baseParams[i]->type == LayerType::LOCAL_AVGPOOL)
            {
                delete reinterpret_cast<LocalAvgPoolParams*>(baseParams[i]);
            }
            else if(baseParams[i]->type == LayerType::CONNECTED)
            {
                delete reinterpret_cast<ConnectParams*>(baseParams[i]);
            }
            else if(baseParams[i]->type == LayerType::BATCHNORM)
            {
                delete reinterpret_cast<BatchNormParams*>(baseParams[i]);
            }
            else if(baseParams[i]->type == LayerType::PADDING)
            {
                delete reinterpret_cast<PaddingParams*>(baseParams[i]);
            }

            baseParams[i] = nullptr;
        }

        if(i == (baseParams.size()-1))
        {
            baseParams.clear();
        }
    }
}
}
