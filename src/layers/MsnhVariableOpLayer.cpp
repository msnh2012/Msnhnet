#include "Msnhnet/layers/MsnhVariableOpLayer.h"

namespace Msnhnet
{

VariableOpLayer::VariableOpLayer(const int &batch, const int &width, const int &height, const int &channel, std::vector<int> &inputLayerIndexes, const VariableOpParams::VarOpType &varOpType, const float &constVal)
{
    this->_type              =   LayerType::VARIABLE_OP;

    this->_layerName         =   "VarOp           ";

    this->_batch             =   batch;
    this->_width             =   width;
    this->_height            =   height;
    this->_channel           =   channel;

    this->_outWidth          =   width;
    this->_outHeight         =   height;
    this->_outChannel        =   channel;

    this->_varOpType         =   varOpType;
    this->_constVal          =   constVal;

    this->_inputLayerIndexes =   inputLayerIndexes;

    this->_inputNum  =   width * height * channel;
    this->_outputNum =   this->_outWidth * this->_outHeight * this->_outChannel;
    this->_maxOutputNum  = this->_batch*this->_outputNum;

    this->_layerDetail.append("VarOp   : " + VariableOpParams::getStrFromVarOpType(varOpType));
    char msg[100];

    for (size_t i = 0; i < inputLayerIndexes.size(); ++i)
    {
#ifdef WIN32
        sprintf_s(msg, " %d", inputLayerIndexes[i]);
#else
        sprintf(msg, " %d", inputLayerIndexes[i]);
#endif
        this->_layerDetail.append(msg);
    }

    this->_layerDetail.append("\n");

}

void VariableOpLayer::forward(NetworkState &netState)
{
    if(this->_layerIndex == 0)
    {
        throw Exception(1,"varop layer should not be 0 layer",__FILE__,__LINE__,__FUNCTION__);
    }

    auto st = TimeUtil::startRecord();

    float* layerInput   = netState.getInput();
    float* layerOutput  = nullptr;

    if(this->_memReUse==1)
    {
        layerOutput     = netState.getOutput(); 

        netState.shuffleInOut();

    }
    else
    {
        layerOutput     = this->_output;
    }

    if(this->_varOpType == VariableOpParams::VAR_OP_ADD || this->_varOpType == VariableOpParams::VAR_OP_SUB || this->_varOpType == VariableOpParams::VAR_OP_SUB_INV||
            this->_varOpType == VariableOpParams::VAR_OP_MUL || this->_varOpType == VariableOpParams::VAR_OP_DIV || this->_varOpType == VariableOpParams::VAR_OP_DIV_INV)
    {
        float *inputA   = nullptr;
        float *inputB   = nullptr;
        if(this->_inputLayerIndexes.size() == 2)
        {
            int indexA          =   this->_inputLayerIndexes[0];
            if(indexA == (this->_layerIndex-1)) 

            {
                if(netState.net->layers[static_cast<size_t>(indexA)]->getMemReUse() == 1) 

                {
                    inputA      =   layerInput;
                }
                else

                {
                    inputA      =   netState.input;
                }

            }
            else 

            {
                inputA      =   netState.net->layers[static_cast<size_t>(indexA)]->getOutput();
            }

            int indexB          =   this->_inputLayerIndexes[1];

            if(indexB == (this->_layerIndex-1)) 

            {
                if(netState.net->layers[static_cast<size_t>(indexB)]->getMemReUse() == 1) 

                {
                    inputB      =   layerInput;
                }
                else

                {
                    inputB      =   netState.input;
                }
            }
            else 

            {
                inputB      =   netState.net->layers[static_cast<size_t>(indexB)]->getOutput();
            }

        }
        else
        {
            if(netState.net->layers[this->_layerIndex-1]->getMemReUse()==1)

            {
                inputA          =   layerInput; 

            }
            else
            {
                inputA          =   netState.input;

            }

            int indexB          =   this->_inputLayerIndexes[0];

            if(indexB == (this->_layerIndex-1)) 

            {
                if(netState.net->layers[static_cast<size_t>(indexB)]->getMemReUse() == 1) 

                {
                    inputB      =   layerInput;
                }
                else

                {
                    inputB      =   netState.input;
                }
            }
            else 

            {
                inputB      =   netState.net->layers[static_cast<size_t>(indexB)]->getOutput();
            }
        }

        for (int j = 0; j < this->_batch; ++j)
        {
            int id = (this->_varOpType - VariableOpParams::VAR_OP_ADD);
            Blas::cpuArithmetic(static_cast<Arithmetic>(id), this->_inputNum, inputA + j*this->_inputNum, 1, inputB  + j*this->_inputNum, 1 , layerOutput  + j*this->_inputNum, 1);
        }
    }
    else if(this->_varOpType == VariableOpParams::VAR_OP_ADD_CONST || this->_varOpType == VariableOpParams::VAR_OP_SUB_CONST || this->_varOpType == VariableOpParams::VAR_OP_SUB_CONST_INV||
            this->_varOpType == VariableOpParams::VAR_OP_MUL_CONST || this->_varOpType == VariableOpParams::VAR_OP_DIV_CONST || this->_varOpType == VariableOpParams::VAR_OP_DIV_CONST_INV)
    {

        float *inputA        = nullptr;

        if(netState.net->layers[this->_layerIndex-1]->getMemReUse() == 1)

        {
            inputA           = layerInput;
        }
        else
        {
            inputA           = netState.input;
        }

        for (int j = 0; j < this->_batch; ++j)
        {
            int id = (this->_varOpType - VariableOpParams::VAR_OP_ADD_CONST);
            Blas::cpuArithmetic(static_cast<Arithmetic>(id), this->_inputNum, inputA + j*this->_inputNum, 1, this->_constVal, layerOutput  + j*this->_inputNum, 1);
        }
    }
    else if(this->_varOpType == VariableOpParams::VAR_OP_ABS || this->_varOpType == VariableOpParams::VAR_OP_ACOS || this->_varOpType == VariableOpParams::VAR_OP_ASIN||
            this->_varOpType == VariableOpParams::VAR_OP_ATAN || this->_varOpType == VariableOpParams::VAR_OP_COS||this->_varOpType == VariableOpParams::VAR_OP_COSH ||
            this->_varOpType == VariableOpParams::VAR_OP_SIN || this->_varOpType == VariableOpParams::VAR_OP_SINH||this->_varOpType == VariableOpParams::VAR_OP_TAN ||
            this->_varOpType == VariableOpParams::VAR_OP_TANH || this->_varOpType == VariableOpParams::VAR_OP_EXP||this->_varOpType == VariableOpParams::VAR_OP_POW ||
            this->_varOpType == VariableOpParams::VAR_OP_LOG || this->_varOpType == VariableOpParams::VAR_OP_LOG10||this->_varOpType == VariableOpParams::VAR_OP_SQRT
            )
    {
        float *inputA        = nullptr;

        if(netState.net->layers[this->_layerIndex-1]->getMemReUse() == 1)

        {
            inputA           = layerInput;
        }
        else
        {
            inputA           = netState.input;
        }

        for (int j = 0; j < this->_batch; ++j)
        {
            int id = (this->_varOpType - VariableOpParams::VAR_OP_ABS);
            Blas::cpuScientific(static_cast<Scientific>(id),this->_inputNum, inputA + j*this->_inputNum, 1, this->_constVal, layerOutput  + j*this->_inputNum, 1, BaseLayer::supportAvx);
        }
    }

    this->_forwardTime =   TimeUtil::getElapsedTime(st);

}

void VariableOpLayer::mallocMemory()
{
    if(!this->_memoryMalloced)
    {
        if(!BaseLayer::isPreviewMode)
        {
            if(!BaseLayer::onlyUseGpu) 

            {
                this->_output        =   new float[static_cast<size_t>(this->_outputNum*this->_batch)]();
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

#ifdef USE_GPU
void VariableOpLayer::forwardGPU(NetworkState &netState)
{
    float* layerGpuInput   = netState.getGpuInput();
    float* layerGpuOutput  = nullptr;

    if(this->_memReUse==1)
    {
        layerGpuOutput     = netState.getGpuOutput(); 

        netState.shuffleGpuInOut();

    }
    else
    {
        layerGpuOutput     = this->_gpuOutput;
    }

    if(this->_varOpType == VariableOpParams::VAR_OP_ADD || this->_varOpType == VariableOpParams::VAR_OP_SUB || this->_varOpType == VariableOpParams::VAR_OP_SUB_INV||
            this->_varOpType == VariableOpParams::VAR_OP_MUL || this->_varOpType == VariableOpParams::VAR_OP_DIV || this->_varOpType == VariableOpParams::VAR_OP_DIV_INV)
    {

        float *gpuInputA;
        float *gpuInputB;
        if(this->_inputLayerIndexes.size() == 2)
        {
            int indexA          =   this->_inputLayerIndexes[0];
            if(indexA == (this->_layerIndex-1)) 

            {
                if(netState.net->layers[static_cast<size_t>(indexA)]->getMemReUse() == 1) 

                {
                    gpuInputA      =   layerGpuInput;
                }
                else

                {
                    gpuInputA      =   netState.input;
                }

            }
            else 

            {
                gpuInputA      =   netState.net->layers[static_cast<size_t>(indexA)]->getGpuOutput();
            }

            int indexB          =   this->_inputLayerIndexes[1];

            if(indexB == (this->_layerIndex-1)) 

            {
                if(netState.net->layers[static_cast<size_t>(indexB)]->getMemReUse() == 1) 

                {
                    gpuInputB      =   layerGpuInput;
                }
                else

                {
                    gpuInputB      =   netState.input;
                }
            }
            else 

            {
                gpuInputB      =   netState.net->layers[static_cast<size_t>(indexB)]->getGpuOutput();
            }

        }
        else
        {
            if(netState.net->layers[this->_layerIndex-1]->getMemReUse()==1)

            {
                gpuInputA          =   layerGpuInput; 

            }
            else
            {
                gpuInputA          =   netState.input;

            }

            int indexB          =   this->_inputLayerIndexes[0];

            if(indexB == (this->_layerIndex-1)) 

            {
                if(netState.net->layers[static_cast<size_t>(indexB)]->getMemReUse() == 1) 

                {
                    gpuInputB      =   layerGpuInput;
                }
                else

                {
                    gpuInputB      =   netState.input;
                }
            }
            else 

            {
                gpuInputB      =   netState.net->layers[static_cast<size_t>(indexB)]->getGpuOutput();
            }
        }

        for (int j = 0; j < this->_batch; ++j)
        {
            int id = (this->_varOpType - VariableOpParams::VAR_OP_ADD);
            BlasGPU::gpuArithmetic(static_cast<Arithmetic>(id), this->_inputNum, gpuInputA + j*this->_inputNum, 1, gpuInputB  + j*this->_inputNum, 1 , layerGpuOutput  + j*this->_inputNum, 1);
        }
    }
    else if(this->_varOpType == VariableOpParams::VAR_OP_ADD_CONST || this->_varOpType == VariableOpParams::VAR_OP_SUB_CONST || this->_varOpType == VariableOpParams::VAR_OP_SUB_CONST_INV||
            this->_varOpType == VariableOpParams::VAR_OP_MUL_CONST || this->_varOpType == VariableOpParams::VAR_OP_DIV_CONST || this->_varOpType == VariableOpParams::VAR_OP_DIV_CONST_INV)
    {
        float *gpuInputA        = nullptr;

        if(netState.net->layers[this->_layerIndex-1]->getMemReUse() == 1)

        {
            gpuInputA           = layerGpuInput;
        }
        else
        {
            gpuInputA           = netState.input;
        }

        for (int j = 0; j < this->_batch; ++j)
        {
            int id = (this->_varOpType - VariableOpParams::VAR_OP_ADD_CONST);
            BlasGPU::gpuArithmetic(static_cast<Arithmetic>(id), this->_inputNum, gpuInputA + j*this->_inputNum, 1, this->_constVal, layerGpuOutput  + j*this->_inputNum, 1);
        }
    }
    else if(this->_varOpType == VariableOpParams::VAR_OP_ABS || this->_varOpType == VariableOpParams::VAR_OP_ACOS || this->_varOpType == VariableOpParams::VAR_OP_ASIN||
            this->_varOpType == VariableOpParams::VAR_OP_ATAN || this->_varOpType == VariableOpParams::VAR_OP_COS||this->_varOpType == VariableOpParams::VAR_OP_COSH ||
            this->_varOpType == VariableOpParams::VAR_OP_SIN || this->_varOpType == VariableOpParams::VAR_OP_SINH||this->_varOpType == VariableOpParams::VAR_OP_TAN ||
            this->_varOpType == VariableOpParams::VAR_OP_TANH || this->_varOpType == VariableOpParams::VAR_OP_EXP||this->_varOpType == VariableOpParams::VAR_OP_POW ||
            this->_varOpType == VariableOpParams::VAR_OP_LOG || this->_varOpType == VariableOpParams::VAR_OP_LOG10||this->_varOpType == VariableOpParams::VAR_OP_SQRT
            )
    {
        float *gpuInputA        = nullptr;

        if(netState.net->layers[this->_layerIndex-1]->getMemReUse() == 1)

        {
            gpuInputA           = layerGpuInput;
        }
        else
        {
            gpuInputA           = netState.input;
        }

        for (int j = 0; j < this->_batch; ++j)
        {
            int id = (this->_varOpType - VariableOpParams::VAR_OP_ABS);
            BlasGPU::gpuScientific(static_cast<Scientific>(id),this->_inputNum, gpuInputA + j*this->_inputNum, 1, this->_constVal, layerGpuOutput  + j*this->_inputNum, 1);
        }
    }
}
#endif

std::vector<int> VariableOpLayer::getInputLayerIndexes() const
{
    return _inputLayerIndexes;
}

float VariableOpLayer::getConstVal() const
{
    return _constVal;
}

VariableOpParams::VarOpType VariableOpLayer::getVarOpType() const
{
    return _varOpType;
}

}

