#include "Msnhnet/layers/MsnhVariableOpLayer.h"

namespace Msnhnet
{

VariableOpLayer::VariableOpLayer(const int &batch, std::vector<int> &inputLayerIndexes, std::vector<int> &inputLayerOutputs, const VariableOpParams::VarOpType &varOpType, const float &constVal)
{
    this->_type              =   LayerType::VARIABLE_OP;

    this->_layerName         =   "VarOp           ";

    this->_batch             =   batch;
    this->_varOpType         =   varOpType;
    this->_constVal          =   constVal;

    int mOutputNum           =   0;

    this->_layerDetail.append("VarOp ");
    char msg[100];

    this->_inputLayerIndexes =   inputLayerIndexes;
    this->_inputLayerOutputs =   inputLayerOutputs;

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

    mOutputNum = inputLayerOutputs[0];

    this->_outputNum     =   mOutputNum;
    this->_inputNum      =   mOutputNum;

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

        int indexA              =   this->_inputLayerIndexes[0];
        int indexB              =   this->_inputLayerIndexes[1];

        float *inputA           =   netState.net->layers[static_cast<size_t>(indexA)]->getOutput();
        float *inputB           =   netState.net->layers[static_cast<size_t>(indexB)]->getOutput();

        int inputLayerOutputs   =   this->_inputLayerOutputs[0];

        for (int j = 0; j < this->_batch; ++j)
        {
            switch (this->_varOpType)
            {
            case VariableOpParams::VAR_OP_ADD:
                Blas::cpuArithmetic(Arithmetic::ARITH_ADD, inputLayerOutputs, inputA + j*inputLayerOutputs, 1, inputB  + j*inputLayerOutputs, 1 , this->_output  + j*inputLayerOutputs, 1);
                break;
            case VariableOpParams::VAR_OP_SUB:
                Blas::cpuArithmetic(Arithmetic::ARITH_SUB, inputLayerOutputs, inputA + j*inputLayerOutputs, 1, inputB  + j*inputLayerOutputs, 1 , this->_output  + j*inputLayerOutputs, 1);
                break;
            case VariableOpParams::VAR_OP_SUB_INV:
                Blas::cpuArithmetic(Arithmetic::ARITH_SUB_INV, inputLayerOutputs, inputA + j*inputLayerOutputs, 1, inputB  + j*inputLayerOutputs, 1 , this->_output  + j*inputLayerOutputs, 1);
                break;
            case VariableOpParams::VAR_OP_MUL:
                Blas::cpuArithmetic(Arithmetic::ARITH_MUL, inputLayerOutputs, inputA + j*inputLayerOutputs, 1, inputB  + j*inputLayerOutputs, 1 , this->_output  + j*inputLayerOutputs, 1);
                break;
            case VariableOpParams::VAR_OP_DIV:
                Blas::cpuArithmetic(Arithmetic::ARITH_DIV, inputLayerOutputs, inputA + j*inputLayerOutputs, 1, inputB  + j*inputLayerOutputs, 1 , this->_output  + j*inputLayerOutputs, 1);
                break;
            case VariableOpParams::VAR_OP_DIV_INV:
                Blas::cpuArithmetic(Arithmetic::ARITH_DIV_INV, inputLayerOutputs, inputA + j*inputLayerOutputs, 1, inputB  + j*inputLayerOutputs, 1 , this->_output  + j*inputLayerOutputs, 1);
                break;
            }

        }
    }
    else if(this->_varOpType == VariableOpParams::VAR_OP_ADD_CONST || this->_varOpType == VariableOpParams::VAR_OP_SUB_CONST || this->_varOpType == VariableOpParams::VAR_OP_SUB_CONST_INV||
            this->_varOpType == VariableOpParams::VAR_OP_MUL_CONST || this->_varOpType == VariableOpParams::VAR_OP_DIV_CONST || this->_varOpType == VariableOpParams::VAR_OP_DIV_CONST_INV)
    {

        int indexA              =   this->_inputLayerIndexes[0];
        float *inputA           =   netState.net->layers[static_cast<size_t>(indexA)]->getOutput();
        int inputLayerOutputs   =   this->_inputLayerOutputs[0];

        for (int j = 0; j < this->_batch; ++j)
        {
            switch (this->_varOpType)
            {
            case VariableOpParams::VAR_OP_ADD:
                Blas::cpuArithmetic(Arithmetic::ARITH_ADD, inputLayerOutputs, inputA + j*inputLayerOutputs, 1, this->_constVal, this->_output  + j*inputLayerOutputs, 1);
                break;
            case VariableOpParams::VAR_OP_SUB:
                Blas::cpuArithmetic(Arithmetic::ARITH_SUB, inputLayerOutputs, inputA + j*inputLayerOutputs, 1, this->_constVal, this->_output  + j*inputLayerOutputs, 1);
                break;
            case VariableOpParams::VAR_OP_SUB_INV:
                Blas::cpuArithmetic(Arithmetic::ARITH_SUB_INV, inputLayerOutputs, inputA + j*inputLayerOutputs, 1, this->_constVal, this->_output  + j*inputLayerOutputs, 1);
                break;
            case VariableOpParams::VAR_OP_MUL:
                Blas::cpuArithmetic(Arithmetic::ARITH_MUL, inputLayerOutputs, inputA + j*inputLayerOutputs, 1, this->_constVal, this->_output  + j*inputLayerOutputs, 1);
                break;
            case VariableOpParams::VAR_OP_DIV:
                Blas::cpuArithmetic(Arithmetic::ARITH_DIV, inputLayerOutputs, inputA + j*inputLayerOutputs, 1, this->_constVal, this->_output  + j*inputLayerOutputs, 1);
                break;
            case VariableOpParams::VAR_OP_DIV_INV:
                Blas::cpuArithmetic(Arithmetic::ARITH_DIV_INV, inputLayerOutputs, inputA + j*inputLayerOutputs, 1, this->_constVal, this->_output  + j*inputLayerOutputs, 1);
                break;
            }

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

        int indexA              =   this->_inputLayerIndexes[0];
        int indexB              =   this->_inputLayerIndexes[1];

        float *gpuInputA           =   netState.net->layers[static_cast<size_t>(indexA)]->getGpuOutput();
        float *gpuInputB           =   netState.net->layers[static_cast<size_t>(indexB)]->getGpuOutput();

        int inputLayerOutputs   =   this->_inputLayerOutputs[0];

        for (int j = 0; j < this->_batch; ++j)
        {
            switch (this->_varOpType)
            {
            case VariableOpParams::VAR_OP_ADD:
                BlasGPU::gpuArithmetic(Arithmetic::ARITH_ADD, inputLayerOutputs, gpuInputA + j*inputLayerOutputs, 1, gpuInputB  + j*inputLayerOutputs, 1 , this->_gpuOutput  + j*inputLayerOutputs, 1);
                break;
            case VariableOpParams::VAR_OP_SUB:
                BlasGPU::gpuArithmetic(Arithmetic::ARITH_SUB, inputLayerOutputs, gpuInputA + j*inputLayerOutputs, 1, gpuInputB  + j*inputLayerOutputs, 1 , this->_gpuOutput  + j*inputLayerOutputs, 1);
                break;
            case VariableOpParams::VAR_OP_SUB_INV:
                BlasGPU::gpuArithmetic(Arithmetic::ARITH_SUB_INV, inputLayerOutputs, gpuInputA + j*inputLayerOutputs, 1, gpuInputB  + j*inputLayerOutputs, 1 , this->_gpuOutput  + j*inputLayerOutputs, 1);
                break;
            case VariableOpParams::VAR_OP_MUL:
                BlasGPU::gpuArithmetic(Arithmetic::ARITH_MUL, inputLayerOutputs, gpuInputA + j*inputLayerOutputs, 1, gpuInputB  + j*inputLayerOutputs, 1 , this->_gpuOutput  + j*inputLayerOutputs, 1);
                break;
            case VariableOpParams::VAR_OP_DIV:
                BlasGPU::gpuArithmetic(Arithmetic::ARITH_DIV, inputLayerOutputs, gpuInputA + j*inputLayerOutputs, 1, gpuInputB  + j*inputLayerOutputs, 1 , this->_gpuOutput  + j*inputLayerOutputs, 1);
                break;
            case VariableOpParams::VAR_OP_DIV_INV:
                BlasGPU::gpuArithmetic(Arithmetic::ARITH_DIV_INV, inputLayerOutputs, gpuInputA + j*inputLayerOutputs, 1, gpuInputB  + j*inputLayerOutputs, 1 , this->_gpuOutput  + j*inputLayerOutputs, 1);
                break;
            }

        }
    }
    else if(this->_varOpType == VariableOpParams::VAR_OP_ADD_CONST || this->_varOpType == VariableOpParams::VAR_OP_SUB_CONST || this->_varOpType == VariableOpParams::VAR_OP_SUB_CONST_INV||
            this->_varOpType == VariableOpParams::VAR_OP_MUL_CONST || this->_varOpType == VariableOpParams::VAR_OP_DIV_CONST || this->_varOpType == VariableOpParams::VAR_OP_DIV_CONST_INV)
    {

        int indexA              =   this->_inputLayerIndexes[0];
        float *gpuInputA           =   netState.net->layers[static_cast<size_t>(indexA)]->getGpuOutput();
        int inputLayerOutputs   =   this->_inputLayerOutputs[0];

        for (int j = 0; j < this->_batch; ++j)
        {
            switch (this->_varOpType)
            {
            case VariableOpParams::VAR_OP_ADD:
                BlasGPU::gpuArithmetic(Arithmetic::ARITH_ADD, inputLayerOutputs, gpuInputA + j*inputLayerOutputs, 1, this->_constVal, this->_gpuOutput  + j*inputLayerOutputs, 1);
                break;
            case VariableOpParams::VAR_OP_SUB:
                BlasGPU::gpuArithmetic(Arithmetic::ARITH_SUB, inputLayerOutputs, gpuInputA + j*inputLayerOutputs, 1, this->_constVal, this->_gpuOutput  + j*inputLayerOutputs, 1);
                break;
            case VariableOpParams::VAR_OP_SUB_INV:
                BlasGPU::gpuArithmetic(Arithmetic::ARITH_SUB_INV, inputLayerOutputs, gpuInputA + j*inputLayerOutputs, 1, this->_constVal, this->_gpuOutput  + j*inputLayerOutputs, 1);
                break;
            case VariableOpParams::VAR_OP_MUL:
                BlasGPU::gpuArithmetic(Arithmetic::ARITH_MUL, inputLayerOutputs, gpuInputA + j*inputLayerOutputs, 1, this->_constVal, this->_gpuOutput  + j*inputLayerOutputs, 1);
                break;
            case VariableOpParams::VAR_OP_DIV:
                BlasGPU::gpuArithmetic(Arithmetic::ARITH_DIV, inputLayerOutputs, gpuInputA + j*inputLayerOutputs, 1, this->_constVal, this->_gpuOutput  + j*inputLayerOutputs, 1);
                break;
            case VariableOpParams::VAR_OP_DIV_INV:
                BlasGPU::gpuArithmetic(Arithmetic::ARITH_DIV_INV, inputLayerOutputs, gpuInputA + j*inputLayerOutputs, 1, this->_constVal, this->_gpuOutput  + j*inputLayerOutputs, 1);
                break;
            }

        }
    }
}
#endif

std::vector<int> VariableOpLayer::inputLayerIndexes() const
{
    return _inputLayerIndexes;
}

std::vector<int> VariableOpLayer::inputLayerOutputs() const
{
    return _inputLayerOutputs;
}

float VariableOpLayer::constVal() const
{
    return _constVal;
}

VariableOpParams::VarOpType VariableOpLayer::varOpType() const
{
    return _varOpType;
}

}

