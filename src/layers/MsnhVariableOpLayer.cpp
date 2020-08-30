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

    this->_layerDetail.append("VarOp   : " + VariableOpParams::getStrFromVarOpType(varOpType));
    char msg[100];

    this->_inputLayerIndexes =   inputLayerIndexes;

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

    this->_inputNum  =   width * height * channel;
    this->_outputNum =   this->_outWidth * this->_outHeight * this->_outChannel;

    if(!BaseLayer::isPreviewMode)
    {
        this->_output        =   new float[static_cast<size_t>(this->_outputNum*this->_batch)]();
#ifdef USE_GPU
        this->_gpuOutput         = Cuda::makeCudaArray(this->_output, this->_outputNum * this->_batch);
#endif
    }
}

void VariableOpLayer::forward(NetworkState &netState)
{
    auto st = TimeUtil::startRecord();

    if(this->_varOpType == VariableOpParams::VAR_OP_ADD || this->_varOpType == VariableOpParams::VAR_OP_SUB || this->_varOpType == VariableOpParams::VAR_OP_SUB_INV||
            this->_varOpType == VariableOpParams::VAR_OP_MUL || this->_varOpType == VariableOpParams::VAR_OP_DIV || this->_varOpType == VariableOpParams::VAR_OP_DIV_INV)
    {
        float *inputA;
        float *inputB;
        if(this->_inputLayerIndexes.size() == 2)
        {
            int indexA          =   this->_inputLayerIndexes[0];
            inputA              =   netState.net->layers[static_cast<size_t>(indexA)]->getOutput();
            int indexB          =   this->_inputLayerIndexes[1];
            inputB              =   netState.net->layers[static_cast<size_t>(indexB)]->getOutput();
        }
        else
        {
            inputA              =   netState.input;
            int indexB          =   this->_inputLayerIndexes[0];
            inputB              =   netState.net->layers[static_cast<size_t>(indexB)]->getOutput();
        }

        for (int j = 0; j < this->_batch; ++j)
        {
            int id = (this->_varOpType - VariableOpParams::VAR_OP_ADD);
            Blas::cpuArithmetic(static_cast<Arithmetic>(id), this->_inputNum, inputA + j*this->_inputNum, 1, inputB  + j*this->_inputNum, 1 , this->_output  + j*this->_inputNum, 1);
        }
    }
    else if(this->_varOpType == VariableOpParams::VAR_OP_ADD_CONST || this->_varOpType == VariableOpParams::VAR_OP_SUB_CONST || this->_varOpType == VariableOpParams::VAR_OP_SUB_CONST_INV||
            this->_varOpType == VariableOpParams::VAR_OP_MUL_CONST || this->_varOpType == VariableOpParams::VAR_OP_DIV_CONST || this->_varOpType == VariableOpParams::VAR_OP_DIV_CONST_INV)
    {

        float *inputA           =   netState.input;

        for (int j = 0; j < this->_batch; ++j)
        {
            int id = (this->_varOpType - VariableOpParams::VAR_OP_ADD_CONST);
            Blas::cpuArithmetic(static_cast<Arithmetic>(id), this->_inputNum, inputA + j*this->_inputNum, 1, this->_constVal, this->_output  + j*this->_inputNum, 1);
        }
    }
    else if(this->_varOpType == VariableOpParams::VAR_OP_ABS || this->_varOpType == VariableOpParams::VAR_OP_ACOS || this->_varOpType == VariableOpParams::VAR_OP_ASIN||
            this->_varOpType == VariableOpParams::VAR_OP_ATAN || this->_varOpType == VariableOpParams::VAR_OP_COS||this->_varOpType == VariableOpParams::VAR_OP_COSH ||
            this->_varOpType == VariableOpParams::VAR_OP_SIN || this->_varOpType == VariableOpParams::VAR_OP_SINH||this->_varOpType == VariableOpParams::VAR_OP_TAN ||
            this->_varOpType == VariableOpParams::VAR_OP_TANH || this->_varOpType == VariableOpParams::VAR_OP_EXP||this->_varOpType == VariableOpParams::VAR_OP_POW ||
            this->_varOpType == VariableOpParams::VAR_OP_LOG || this->_varOpType == VariableOpParams::VAR_OP_LOG10||this->_varOpType == VariableOpParams::VAR_OP_SQRT
            )
    {
        float *inputA           =   netState.input;

        for (int j = 0; j < this->_batch; ++j)
        {
            int id = (this->_varOpType - VariableOpParams::VAR_OP_ABS);
            Blas::cpuScientific(static_cast<Scientific>(id),this->_inputNum, inputA + j*this->_inputNum, 1, this->_constVal, this->_output  + j*this->_inputNum, 1, BaseLayer::supportAvx);
        }
    }

    this->_forwardTime =   TimeUtil::getElapsedTime(st);

}

#ifdef USE_GPU
void VariableOpLayer::forwardGPU(NetworkState &netState)
{
    if(this->_varOpType == VariableOpParams::VAR_OP_ADD || this->_varOpType == VariableOpParams::VAR_OP_SUB || this->_varOpType == VariableOpParams::VAR_OP_SUB_INV||
            this->_varOpType == VariableOpParams::VAR_OP_MUL || this->_varOpType == VariableOpParams::VAR_OP_DIV || this->_varOpType == VariableOpParams::VAR_OP_DIV_INV)
    {

        float *gpuInputA;
        float *gpuInputB;
        if(this->_inputLayerIndexes.size() == 2)
        {
            int indexA          =   this->_inputLayerIndexes[0];
            gpuInputA           =   netState.net->layers[static_cast<size_t>(indexA)]->getGpuOutput();
            int indexB          =   this->_inputLayerIndexes[1];
            gpuInputB           =   netState.net->layers[static_cast<size_t>(indexB)]->getGpuOutput();
        }
        else
        {
            gpuInputA           =   netState.input;
            int indexB          =   this->_inputLayerIndexes[0];
            gpuInputB           =   netState.net->layers[static_cast<size_t>(indexB)]->getGpuOutput();
        }

        for (int j = 0; j < this->_batch; ++j)
        {
            int id = (this->_varOpType - VariableOpParams::VAR_OP_ADD);
            BlasGPU::gpuArithmetic(static_cast<Arithmetic>(id), this->_inputNum, gpuInputA + j*this->_inputNum, 1, gpuInputB  + j*this->_inputNum, 1 , this->_gpuOutput  + j*this->_inputNum, 1);
        }
    }
    else if(this->_varOpType == VariableOpParams::VAR_OP_ADD_CONST || this->_varOpType == VariableOpParams::VAR_OP_SUB_CONST || this->_varOpType == VariableOpParams::VAR_OP_SUB_CONST_INV||
            this->_varOpType == VariableOpParams::VAR_OP_MUL_CONST || this->_varOpType == VariableOpParams::VAR_OP_DIV_CONST || this->_varOpType == VariableOpParams::VAR_OP_DIV_CONST_INV)
    {
        float *gpuInputA           =   netState.input;

        for (int j = 0; j < this->_batch; ++j)
        {
            int id = (this->_varOpType - VariableOpParams::VAR_OP_ADD_CONST);
            BlasGPU::gpuArithmetic(static_cast<Arithmetic>(id), this->_inputNum, gpuInputA + j*this->_inputNum, 1, this->_constVal, this->_gpuOutput  + j*this->_inputNum, 1);
        }
    }
    else if(this->_varOpType == VariableOpParams::VAR_OP_ABS || this->_varOpType == VariableOpParams::VAR_OP_ACOS || this->_varOpType == VariableOpParams::VAR_OP_ASIN||
            this->_varOpType == VariableOpParams::VAR_OP_ATAN || this->_varOpType == VariableOpParams::VAR_OP_COS||this->_varOpType == VariableOpParams::VAR_OP_COSH ||
            this->_varOpType == VariableOpParams::VAR_OP_SIN || this->_varOpType == VariableOpParams::VAR_OP_SINH||this->_varOpType == VariableOpParams::VAR_OP_TAN ||
            this->_varOpType == VariableOpParams::VAR_OP_TANH || this->_varOpType == VariableOpParams::VAR_OP_EXP||this->_varOpType == VariableOpParams::VAR_OP_POW ||
            this->_varOpType == VariableOpParams::VAR_OP_LOG || this->_varOpType == VariableOpParams::VAR_OP_LOG10||this->_varOpType == VariableOpParams::VAR_OP_SQRT
            )
    {
        float *inputA           =   netState.input;

        for (int j = 0; j < this->_batch; ++j)
        {
            int id = (this->_varOpType - VariableOpParams::VAR_OP_ABS);
            BlasGPU::gpuScientific(static_cast<Scientific>(id),this->_inputNum, inputA + j*this->_inputNum, 1, this->_constVal, this->_output  + j*this->_inputNum, 1);
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

