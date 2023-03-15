//
// Copyright © 2022 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "Converter.hpp"
#include <half/half.hpp>
#include <armnnUtils/TensorUtils.hpp>

namespace armnn_driver
{

using namespace android::nn;
using Half = half_float::half;

namespace
{

} // anonymouse namespace

bool Converter::ConvertOperation(const Operation& operation, const Model& model, ConversionData& data)
{
    switch (operation.type)
    {
        case OperationType::ABS:
            return ConvertElementwiseUnary(operation, model, data, UnaryOperation::Abs);
        case OperationType::ADD:
            return ConvertAdd(operation, model, data);
        case OperationType::ARGMAX:
            return ConvertArgMinMax(operation, model, data, ArgMinMaxFunction::Max);
        case OperationType::ARGMIN:
            return ConvertArgMinMax(operation, model, data, ArgMinMaxFunction::Min);
        case OperationType::AVERAGE_POOL_2D:
            return ConvertAveragePool2d(operation, model, data);
        case OperationType::BATCH_MATMUL:
            return ConvertBatchMatMul(operation, model, data);
        case OperationType::BATCH_TO_SPACE_ND:
            return ConvertBatchToSpaceNd(operation, model, data);
        case OperationType::CAST:
            return ConvertCast(operation, model, data);
        case OperationType::CONCATENATION:
            return ConvertConcatenation(operation, model, data);
        case OperationType::CONV_2D:
            return ConvertConv2d(operation, model, data);
        case OperationType::DEPTH_TO_SPACE:
            return ConvertDepthToSpace(operation, model, data);
        case OperationType::DEPTHWISE_CONV_2D:
            return ConvertDepthwiseConv2d(operation, model, data);
        case OperationType::DEQUANTIZE:
            return ConvertDequantize(operation, model, data);
        case OperationType::DIV:
            return ConvertDiv(operation, model, data);
        case OperationType::ELU:
            return ConvertElu(operation, model, data);
        case OperationType::EQUAL:
            return ConvertComparison(operation, model, data, ComparisonOperation::Equal);
        case OperationType::EXP:
            return ConvertElementwiseUnary(operation, model, data, UnaryOperation::Exp);
        case OperationType::EXPAND_DIMS:
            return ConvertExpandDims(operation, model, data);
        case OperationType::FILL:
            return ConvertFill(operation, model, data);
        case OperationType::FLOOR:
            return ConvertFloor(operation, model, data);
        case OperationType::FULLY_CONNECTED:
            return ConvertFullyConnected(operation, model, data);
        case OperationType::GATHER:
            return ConvertGather(operation, model, data);
        case OperationType::GREATER:
            return ConvertComparison(operation, model, data, ComparisonOperation::Greater);
        case OperationType::GREATER_EQUAL:
            return ConvertComparison(operation, model, data, ComparisonOperation::GreaterOrEqual);
        case OperationType::GROUPED_CONV_2D:
            return ConvertGroupedConv2d(operation, model, data);
        case OperationType::HARD_SWISH:
            return ConvertHardSwish(operation, model, data);
        case OperationType::INSTANCE_NORMALIZATION:
            return ConvertInstanceNormalization(operation, model, data);
        case OperationType::L2_NORMALIZATION:
            return ConvertL2Normalization(operation, model, data);
        case OperationType::L2_POOL_2D:
            return ConvertL2Pool2d(operation, model, data);
        case OperationType::LESS:
            return ConvertComparison(operation, model, data, ComparisonOperation::Less);
        case OperationType::LESS_EQUAL:
            return ConvertComparison(operation, model, data, ComparisonOperation::LessOrEqual);
        case OperationType::LOCAL_RESPONSE_NORMALIZATION:
            return ConvertLocalResponseNormalization(operation, model, data);
        case OperationType::LOG:
            return ConvertElementwiseUnary(operation, model, data, UnaryOperation::Log);
        case OperationType::LOGICAL_AND:
            return ConvertLogicalBinary(operation, model, data, LogicalBinaryOperation::LogicalAnd);
        case OperationType::LOGICAL_NOT:
            return ConvertElementwiseUnary(operation, model, data, UnaryOperation::LogicalNot);
        case OperationType::LOGICAL_OR:
            return ConvertLogicalBinary(operation, model, data, LogicalBinaryOperation::LogicalOr);
        case OperationType::LOGISTIC:
            return ConvertLogistic(operation, model, data);
        case OperationType::LOG_SOFTMAX:
            return ConvertLogSoftmax(operation, model, data);
        case OperationType::LSTM:
            return ConvertLstm(operation, model, data);
        case OperationType::MAX_POOL_2D:
            return ConvertMaxPool2d(operation, model, data);
        case OperationType::MAXIMUM:
            return ConvertMaximum(operation, model, data);
        case OperationType::MEAN:
            return ConvertMean(operation, model, data);
        case OperationType::MINIMUM:
            return ConvertMinimum(operation, model, data);
        case OperationType::MUL:
            return ConvertMul(operation, model, data);
        case OperationType::NEG:
            return ConvertElementwiseUnary(operation, model, data, UnaryOperation::Neg);
        case OperationType::NOT_EQUAL:
            return ConvertComparison(operation, model, data, ComparisonOperation::NotEqual);
        case OperationType::PAD:
            return ConvertPad(operation, model, data);
        case OperationType::PAD_V2:
            return ConvertPadV2(operation, model, data);
        case OperationType::PRELU:
            return ConvertPrelu(operation, model, data);
        case OperationType::QUANTIZE:
            return ConvertQuantize(operation, model, data);
        case OperationType::QUANTIZED_LSTM:
            return ConvertQuantizedLstm(operation, model, data);
        case OperationType::QUANTIZED_16BIT_LSTM:
            return ConvertQuantized16BitLstm(operation, model, data);
        case OperationType::RANK:
            return ConvertRank(operation, model, data);
        case OperationType::REDUCE_MAX:
            return ConvertReduce(operation, model, data, armnn::ReduceOperation::Max);
        case OperationType::REDUCE_MIN:
            return ConvertReduce(operation, model, data, armnn::ReduceOperation::Min);
        case OperationType::REDUCE_SUM:
            return ConvertReduce(operation, model, data, armnn::ReduceOperation::Sum);
        case OperationType::RELU:
            return ConvertReLu(operation, model, data);
        case OperationType::RELU1:
            return ConvertReLu1(operation, model, data);
        case OperationType::RELU6:
            return ConvertReLu6(operation, model, data);
        case OperationType::RESHAPE:
            return ConvertReshape(operation, model, data);
        case OperationType::RESIZE_BILINEAR:
            return ConvertResize(operation, model, data, ResizeMethod::Bilinear);
        case OperationType::RESIZE_NEAREST_NEIGHBOR:
            return ConvertResize(operation, model, data, ResizeMethod::NearestNeighbor);
        case OperationType::RSQRT:
            return ConvertElementwiseUnary(operation, model, data, UnaryOperation::Rsqrt);
        case OperationType::SIN:
            return ConvertElementwiseUnary(operation, model, data, UnaryOperation::Sin);
        case OperationType::SOFTMAX:
            return ConvertSoftmax(operation, model, data);
        case OperationType::SPACE_TO_BATCH_ND  :
            return ConvertSpaceToBatchNd(operation, model, data);
        case OperationType::SPACE_TO_DEPTH:
            return ConvertSpaceToDepth(operation, model, data);
        case OperationType::SQRT:
            return ConvertSqrt(operation, model, data);
        case OperationType::SQUEEZE:
            return ConvertSqueeze(operation, model, data);
        case OperationType::STRIDED_SLICE:
            return ConvertStridedSlice(operation, model, data);
        case OperationType::SUB:
            return ConvertSub(operation, model, data);
        case OperationType::TRANSPOSE:
            return ConvertTranspose(operation, model, data);
        case OperationType::TRANSPOSE_CONV_2D:
            return ConvertTransposeConv2d(operation, model, data);
        case OperationType::TANH:
            return ConvertTanH(operation, model, data);
        default:
            VLOG(DRIVER) << "Operation type: " << operation.type << "is not supported in ArmnnDriver";
            return false;
    }
}

bool Converter::ConvertAdd(const Operation& operation, const Model& model, ConversionData& data)
{
    VLOG(DRIVER) << "Converter::ConvertAdd()";
    LayerInputHandle input0 = ConvertToLayerInputHandle(operation, 0, model, data);
    LayerInputHandle input1 = ConvertToLayerInputHandle(operation, 1, model, data);

    if (!input0.IsValid() || !input1.IsValid())
    {
        return Fail("%s: Operation has invalid inputs", __func__);
    }

    // The FuseActivation parameter is always the input index 2, and it should be optional
    ActivationFn activationFunction;
    if (!GetOptionalInputActivation(operation, 2, activationFunction, model, data))
    {
        return Fail("%s: Operation has invalid inputs", __func__);
    }

    const Operand* outputOperand = GetOutputOperand(operation, 0, model);
    if (!outputOperand)
    {
        return false;
    }

    const armnn::TensorInfo& inputInfo0 = input0.GetTensorInfo();
    const armnn::TensorInfo& inputInfo1 = input1.GetTensorInfo();

    const armnn::TensorInfo& outputInfo = GetTensorInfoForOperand(*outputOperand);

    bool isSupported = false;
    armnn::BackendId setBackend;
    auto validateFunc = [&](const armnn::TensorInfo& outputInfo, bool& isSupported)
    {
        ARMNN_NO_DEPRECATE_WARN_BEGIN
        FORWARD_LAYER_SUPPORT_FUNC(__func__,
                                   IsAdditionSupported,
                                   data.m_Backends,
                                   isSupported,
                                   setBackend,
                                   inputInfo0,
                                   inputInfo1,
                                   outputInfo);
        ARMNN_NO_DEPRECATE_WARN_END
    };

    if(!IsDynamicTensor(outputInfo))
    {
        validateFunc(outputInfo, isSupported);
    }
    else
    {
        isSupported = AreDynamicTensorsSupported();
    }

    if (!isSupported)
    {
        return false;
    }

    ARMNN_NO_DEPRECATE_WARN_BEGIN
    armnn::IConnectableLayer* const startLayer = data.m_Network->AddAdditionLayer();
    ARMNN_NO_DEPRECATE_WARN_END
    startLayer->SetBackendId(setBackend);

    bool isReshapeSupported = BroadcastTensor(input0, input1, startLayer, data);
    if (!isReshapeSupported)
    {
        return false;
    }

    return SetupAndTrackLayerOutputSlot(operation, 0, *startLayer, model,
                                        data, nullptr, validateFunc, activationFunction);
}

bool Converter::ConvertArgMinMax(const Operation& operation,
                                 const Model& model,
                                 ConversionData& data,
                                 armnn::ArgMinMaxFunction argMinMaxFunction)
{
    VLOG(DRIVER) << "Converter::ConvertArgMinMax()";
    VLOG(DRIVER) << "argMinMaxFunction = " << GetArgMinMaxFunctionAsCString(argMinMaxFunction);

    LayerInputHandle input0 = ConvertToLayerInputHandle(operation, 0, model, data);

    if (!input0.IsValid())
    {
        return Fail("%s: Operation has invalid inputs", __func__);
    }

    int32_t axis;
    if (!GetInputScalar(operation, 1, OperandType::INT32, axis, model, data))
    {
        return Fail("%s: Operation has invalid inputs. Failed to read axis.", __func__);
    }

    const armnn::TensorInfo& inputInfo = input0.GetTensorInfo();
    int rank = static_cast<int>(inputInfo.GetNumDimensions());

    if (((axis < -rank) && (axis < 0)) || ((axis >= rank) && (axis > 0)))
    {
        // Square bracket denotes inclusive n while parenthesis denotes exclusive n
        // E.g. Rank 4 tensor can have axis in range [-4, 3)
        // -1 == 3, -2 == 2, -3 == 1, -4 == 0
        return Fail("%s: Axis must be in range [-n, n)", __func__);
    }

    const Operand* output = GetOutputOperand(operation, 0, model);
    if (!output)
    {
        return Fail("%s: Could not read output 0", __func__);
    }

    const armnn::TensorInfo& inputInfo0 = input0.GetTensorInfo();

    const armnn::TensorInfo& outputInfo = GetTensorInfoForOperand(*output);

    armnn::ArgMinMaxDescriptor descriptor;
    descriptor.m_Function = argMinMaxFunction;
    descriptor.m_Axis     = axis;

    bool isSupported = false;
    armnn::BackendId setBackend;
    auto validateFunc = [&](const armnn::TensorInfo& outputInfo, bool& isSupported)
    {
        FORWARD_LAYER_SUPPORT_FUNC(__func__,
                                   IsArgMinMaxSupported,
                                   data.m_Backends,
                                   isSupported,
                                   setBackend,
                                   inputInfo0,
                                   outputInfo,
                                   descriptor);
    };

    if(IsDynamicTensor(outputInfo))
    {
        isSupported = AreDynamicTensorsSupported();
    }
    else
    {
        validateFunc(outputInfo, isSupported);
    }

    if (!isSupported)
    {
        return false;
    }

    armnn::IConnectableLayer* layer = data.m_Network->AddArgMinMaxLayer(descriptor);
    layer->SetBackendId(setBackend);
    assert(layer != nullptr);

    input0.Connect(layer->GetInputSlot(0));

    return SetupAndTrackLayerOutputSlot(operation, 0, *layer, model, data, nullptr, validateFunc);
}

bool Converter::ConvertAveragePool2d(const Operation& operation, const Model& model, ConversionData& data)
{
    VLOG(DRIVER) << "Converter::ConvertAveragePool2d()";
    return ConvertPooling2d(operation, __func__, PoolingAlgorithm::Average, model, data);
}

bool Converter::ConvertBatchMatMul(const Operation& operation, const Model& model, ConversionData& data)
{
    VLOG(DRIVER) << "Converter::ConvertBatchMatMul()";
    LayerInputHandle input0 = ConvertToLayerInputHandle(operation, 0, model, data);
    LayerInputHandle input1 = ConvertToLayerInputHandle(operation, 1, model, data);

    if (!input0.IsValid() || !input1.IsValid())
    {
        return Fail("%s: Operation has invalid inputs", __func__);
    }

    const armnn::TensorInfo& inputInfo0 = input0.GetTensorInfo();
    const armnn::TensorInfo& inputInfo1 = input1.GetTensorInfo();

    unsigned int rankInput0 = inputInfo0.GetNumDimensions();
    if (rankInput0 > 4 || rankInput0 < 2)
    {
        Fail("%s: Only inputs with rank at least 2 and up to 4 are supported", __func__);
    }

    unsigned int rankInput1 = inputInfo1.GetNumDimensions();
    if (rankInput1 > 4 || rankInput1 < 2)
    {
        Fail("%s: Only inputs with rank at least 2 and up to 4 are supported", __func__);
    }

    // Determine data type of input tensor 0
    OperandType input0Type;
    if (!GetOperandType(operation, 0, model, input0Type))
    {
        return Fail("%s: Operation has invalid inputs", __func__);
    }

    // Determine data type of input tensor 0
    OperandType input1Type;
    if (!GetOperandType(operation, 0, model, input1Type))
    {
        return Fail("%s: Operation has invalid inputs", __func__);
    }

    if (input0Type != input1Type)
    {
        return Fail("%s: Operation has invalid inputs (Inputs must have same OperandCode)", __func__);
    }

    const Operand* output = GetOutputOperand(operation, 0, model);
    if (!output)
    {
        return Fail("%s: Could not read output 0", __func__);
    }

    const armnn::TensorInfo& outputInfo = GetTensorInfoForOperand(*output);

    armnn::BatchMatMulDescriptor batchMatMulDesc;

    // Inputs 2 and 3 are adjoint in Android NeuralNetworks, but they perform transpose.
    // This is why we are linking them with transpose parameters in the descriptor
    batchMatMulDesc.m_TransposeX = GetOptionalBool(operation, 2, model, data);
    batchMatMulDesc.m_TransposeY = GetOptionalBool(operation, 3, model, data);

    bool isSupported = false;
    armnn::BackendId setBackend;
    auto validateFunc = [&](const armnn::TensorInfo& outputInfo, bool& isSupported)
    {
        FORWARD_LAYER_SUPPORT_FUNC(__func__,
                                   IsBatchMatMulSupported,
                                   data.m_Backends,
                                   isSupported,
                                   setBackend,
                                   inputInfo0,
                                   inputInfo1,
                                   outputInfo,
                                   batchMatMulDesc);
    };

    if(!IsDynamicTensor(outputInfo))
    {
        validateFunc(outputInfo, isSupported);
    }
    else
    {
        isSupported = AreDynamicTensorsSupported();
    }


    if (!isSupported)
    {
        return false;
    }

    armnn::IConnectableLayer* const layer = data.m_Network->AddBatchMatMulLayer(batchMatMulDesc);
    layer->SetBackendId(setBackend);
    assert(layer != nullptr);
    input0.Connect(layer->GetInputSlot(0));
    input1.Connect(layer->GetInputSlot(1));

    return SetupAndTrackLayerOutputSlot(operation, 0, *layer, model, data, nullptr, validateFunc);
}

bool Converter::ConvertBatchToSpaceNd(const Operation& operation, const Model& model, ConversionData& data)
{
    VLOG(DRIVER) << "Converter::ConvertBatchToSpaceNd()";
    LayerInputHandle input = ConvertToLayerInputHandle(operation, 0, model, data);
    if (!input.IsValid())
    {
        return Fail("%s: Operation has invalid inputs", __func__);
    }

    const Operand* output = GetOutputOperand(operation, 0, model);
    if (!output)
    {
        return Fail("%s: Could not read output 0", __func__);
    }

    const armnn::TensorInfo& outputInfo = GetTensorInfoForOperand(*output);

    const Operand* blockOperand = GetInputOperand(operation, 1, model);
    if (!blockOperand)
    {
        return Fail("%s: Could not read input 1", __func__);
    }

    // Convert the block operand to int32
    std::vector<int32_t> block;
    if (!GetTensorInt32Values(*blockOperand, block, model, data))
    {
        return Fail("%s: Input 1 has invalid values", __func__);
    }

    const armnn::TensorInfo& inputInfo = input.GetTensorInfo();

    unsigned int rank = inputInfo.GetNumDimensions();
    if (rank != 4)
    {
        Fail("%s: Only inputs with rank equal to 4 are supported", __func__);
    }

    if (std::any_of(block.cbegin(), block.cend(), [](int32_t i){ return i < 1; }))
    {
        return Fail("%s: Block sizes for each spatial dimension of the input tensor must be"
                    " greater than or equal to 1", __func__);
    }

    armnn::BatchToSpaceNdDescriptor batchToSpaceNdDesc;
    batchToSpaceNdDesc.m_BlockShape.assign(block.cbegin(), block.cend());
    batchToSpaceNdDesc.m_DataLayout = armnn::DataLayout::NHWC;

    if (Is12OrLaterOperand(*output))
    {
        batchToSpaceNdDesc.m_DataLayout = OptionalDataLayout(operation, 2, model, data);
    }
    // Setting crops to 0,0 0,0 as it is not supported in Android NN API
    batchToSpaceNdDesc.m_Crops = {{0, 0}, {0, 0}};

    bool isSupported = false;
    armnn::BackendId setBackend;
    auto validateFunc = [&](const armnn::TensorInfo& outputInfo, bool& isSupported)
    {
        FORWARD_LAYER_SUPPORT_FUNC(__func__,
                                   IsBatchToSpaceNdSupported,
                                   data.m_Backends,
                                   isSupported,
                                   setBackend,
                                   inputInfo,
                                   outputInfo,
                                   batchToSpaceNdDesc);
    };

    if(!IsDynamicTensor(outputInfo))
    {
        validateFunc(outputInfo, isSupported);
    }
    else
    {
        isSupported = AreDynamicTensorsSupported();
    }


    if (!isSupported)
    {
        return false;
    }

    armnn::IConnectableLayer* const layer = data.m_Network->AddBatchToSpaceNdLayer(batchToSpaceNdDesc);
    layer->SetBackendId(setBackend);
    assert(layer != nullptr);
    input.Connect(layer->GetInputSlot(0));

    return SetupAndTrackLayerOutputSlot(operation, 0, *layer, model, data, nullptr, validateFunc);
}

bool Converter::ConvertCast(const Operation& operation, const Model& model, ConversionData& data)
{
    VLOG(DRIVER) << "Converter::ConvertCast()";

    LayerInputHandle input = ConvertToLayerInputHandle(operation, 0, model, data);

    if (!input.IsValid())
    {
        return Fail("%s: Operation has invalid inputs", __func__);
    }

    const Operand* output = GetOutputOperand(operation, 0, model);
    if (!output)
    {
        return Fail("%s: Could not read output 0", __func__);
    }

    const TensorInfo& inputInfo  = input.GetTensorInfo();
    const TensorInfo& outputInfo = GetTensorInfoForOperand(*output);

    bool isSupported = false;
    armnn::BackendId setBackend;
    auto validateFunc = [&](const armnn::TensorInfo& outputInfo, bool& isSupported)
    {
        FORWARD_LAYER_SUPPORT_FUNC(__func__,
                                   IsCastSupported,
                                   data.m_Backends,
                                   isSupported,
                                   setBackend,
                                   inputInfo,
                                   outputInfo);
    };

    if(!IsDynamicTensor(outputInfo))
    {
        validateFunc(outputInfo, isSupported);
    }
    else
    {
        isSupported = AreDynamicTensorsSupported();
    }

    if (!isSupported)
    {
        return false;
    }

    IConnectableLayer* layer = data.m_Network->AddCastLayer();
    layer->SetBackendId(setBackend);
    assert(layer != nullptr);
    input.Connect(layer->GetInputSlot(0));

    return SetupAndTrackLayerOutputSlot(operation, 0, *layer, model, data, nullptr, validateFunc);
}

bool Converter::ConvertComparison(const Operation& operation,
                                  const Model& model,
                                  ConversionData& data,
                                  ComparisonOperation comparisonOperation)
{
    VLOG(DRIVER) << "Converter::ConvertComparison()";
    VLOG(DRIVER) << "comparisonOperation = " << GetComparisonOperationAsCString(comparisonOperation);

    LayerInputHandle input0 = ConvertToLayerInputHandle(operation, 0, model, data);
    LayerInputHandle input1 = ConvertToLayerInputHandle(operation, 1, model, data);

    if (!(input0.IsValid() && input1.IsValid()))
    {
        return Fail("%s: Operation has invalid inputs", __func__);
    }

    const Operand* output = GetOutputOperand(operation, 0, model);
    if (!output)
    {
        return Fail("%s: Could not read output 0", __func__);
    }

    const TensorInfo& inputInfo0 = input0.GetTensorInfo();
    const TensorInfo& inputInfo1 = input1.GetTensorInfo();
    const TensorInfo& outputInfo = GetTensorInfoForOperand(*output);

    ComparisonDescriptor descriptor(comparisonOperation);

    bool isSupported = false;
    armnn::BackendId setBackend;
    auto validateFunc = [&](const armnn::TensorInfo& outputInfo, bool& isSupported)
    {
        FORWARD_LAYER_SUPPORT_FUNC(__func__,
                                   IsComparisonSupported,
                                   data.m_Backends,
                                   isSupported,
                                   setBackend,
                                   inputInfo0,
                                   inputInfo1,
                                   outputInfo,
                                   descriptor);
    };

    if(!IsDynamicTensor(outputInfo))
    {
        validateFunc(outputInfo, isSupported);
    }
    else
    {
        isSupported = AreDynamicTensorsSupported();
    }

    if (!isSupported)
    {
        return false;
    }

    IConnectableLayer* layer = data.m_Network->AddComparisonLayer(descriptor);
    layer->SetBackendId(setBackend);
    assert(layer != nullptr);

    bool isReshapeSupported = BroadcastTensor(input0, input1, layer, data);
    if (!isReshapeSupported)
    {
        return false;
    }

    if(IsDynamicTensor(outputInfo))
    {
        input0.Connect(layer->GetInputSlot(0));
        input1.Connect(layer->GetInputSlot(1));
    }

    return SetupAndTrackLayerOutputSlot(operation, 0, *layer, model, data, nullptr, validateFunc);
}


bool Converter::ConvertConcatenation(const Operation& operation, const Model& model, ConversionData& data)
{
    VLOG(DRIVER) << "Converter::ConvertConcatenation()";

    // The first N (0..N-1) inputs are tensors. The Nth input is the concatenation axis.
    if (operation.inputs.size() <= 1)
    {
        return Fail("%s: Operation has insufficient arguments", __func__);
    }

    // Get inputs and outputs
    const std::size_t numInputTensors = operation.inputs.size() - 1;

    int32_t concatDim;
    if (!GetInputScalar(operation, numInputTensors, OperandType::INT32, concatDim, model, data))
    {
        return Fail("%s: Operation has invalid inputs", __func__);
    }

    const Operand* outputOperand = GetOutputOperand(operation, 0, model);
    if (!outputOperand)
    {
        return Fail("%s: Operation has no outputs", __func__);
    }

    armnn::TensorInfo  outputInfo      = GetTensorInfoForOperand(*outputOperand);
    armnn::TensorShape outputShape     = outputInfo.GetShape();
    const bool         isDynamicTensor = IsDynamicTensor(outputInfo);
    //
    // handle negative concat dims along the lines of tensorflow as described here:
    //    https://www.tensorflow.org/api_docs/python/tf/concat
    // "negative axis refers to axis + rank(values)-th dimension"
    //
    if (concatDim < 0)
    {
        concatDim += outputShape.GetNumDimensions();
    }

    if (concatDim >= static_cast<int32_t>(outputShape.GetNumDimensions()) || concatDim < 0)
    {
        return Fail("%s: Operation has invalid concat axis: %d", __func__, concatDim);
    }

    std::vector<LayerInputHandle>   inputHandles;
    std::vector<armnn::TensorShape> inputShapes;

    inputHandles.reserve(numInputTensors);
    inputShapes.reserve(numInputTensors);

    bool          inputsHaveBeenReshaped = false;
    unsigned int  tensorDimensionsAdded  = 0;
    for (uint32_t i = 0; i < numInputTensors; ++i)
    {
        const Operand* operand = GetInputOperand(operation, i, model);
        if (!operand)
        {
            return Fail("%s: Operation has invalid inputs", __func__);
        }

        LayerInputHandle operandInputHandle = ConvertToLayerInputHandle(operation, i, model, data);
        if (!operandInputHandle.IsValid())
        {
            return Fail("%s: Operation has invalid inputs", __func__);
        }

        armnn::TensorShape operandShape = GetTensorShapeForOperand(*operand);
        if (operandShape.GetNumDimensions() == 0)
        {
            return Fail("%s: Operands with rank 0 are not supported", __func__);
        }

        if (RequiresReshape(operandShape))
        {
            inputsHaveBeenReshaped = true;

            armnn::TensorInfo reshapeInfo = operandInputHandle.GetTensorInfo();

            // Expand the tensor to three dimensions
            if (operandShape.GetNumDimensions() == 2)
            {
                reshapeInfo.SetShape(armnn::TensorShape({1, operandShape[0], operandShape[1]}));
                tensorDimensionsAdded = 1;
            }
            else
            {
                reshapeInfo.SetShape(armnn::TensorShape({1, 1, operandShape[0]}));
                tensorDimensionsAdded = 2;
            }

            armnn::ReshapeDescriptor reshapeDescriptor;
            reshapeDescriptor.m_TargetShape = reshapeInfo.GetShape();

            bool isSupported = false;
            armnn::BackendId setBackendReshape;
            FORWARD_LAYER_SUPPORT_FUNC(__func__,
                                       IsReshapeSupported,
                                       data.m_Backends,
                                       isSupported,
                                       setBackendReshape,
                                       operandInputHandle.GetTensorInfo(),
                                       reshapeInfo,
                                       reshapeDescriptor);

            if (!isSupported)
            {
                return false;
            }
            armnn::IConnectableLayer& newReshape = AddReshapeLayer(*data.m_Network, operandInputHandle, reshapeInfo);
            newReshape.SetBackendId(setBackendReshape);

            // Point to the reshape operation rather then the input operation
            operandShape       = reshapeInfo.GetShape();
            operandInputHandle = LayerInputHandle(true, &newReshape.GetOutputSlot(0), reshapeInfo);
        }

        inputShapes.emplace_back(operandShape);
        inputHandles.emplace_back(operandInputHandle);

        if (!inputHandles.back().IsValid())
        {
            return Fail("%s: Operation has invalid inputs", __func__);
        }
    }

    ARMNN_ASSERT(inputShapes.size() == inputHandles.size());

    if (inputsHaveBeenReshaped)
    {
        // Adjust the concatenation dimension by the amount of dimensions added (if any)
        concatDim += tensorDimensionsAdded;

        // Add extra dimensions to the output shape to reflect the addition of the reshape layers
        if (tensorDimensionsAdded == 1)
        {
            if (IsDynamicTensor(outputInfo))
            {
                outputShape = armnn::TensorShape({1, 0, 0}, {true, false, false});
            }
            else
            {
                outputShape = armnn::TensorShape({1, outputShape[0], outputShape[1]});
            }
        }
        else if (tensorDimensionsAdded == 2)
        {
            if (IsDynamicTensor(outputInfo))
            {
                outputShape = armnn::TensorShape({1, 1, 0}, {true, true, false});
            }
            else
            {
                outputShape = armnn::TensorShape({1, 1, outputShape[0]});
            }
        }
    }

    // Check if permutations is required and get the pair of permutations required for the concatenation.
    // Permutation is required when the concat dimension is 2 for a 4D tensor or 1 for a 3D tensor.
    std::pair<armnn::PermutationVector, armnn::PermutationVector> permutationPair =
            std::make_pair(IdentityPermutation4D, IdentityPermutation4D);
    bool needPermute = CreateConcatPermutationParameters(inputShapes[0].GetNumDimensions(),
                                                         concatDim,
                                                         permutationPair);

    // Only relevant to static tensors as dynamic output tensors will be transposed as a result of inferring from input
    if (!isDynamicTensor)
    {
        if (needPermute)
        {
            outputShape = armnnUtils::TransposeTensorShape(outputShape, permutationPair.first);
        }

        outputInfo.SetShape(outputShape);
    }
    // this is no-op for identity swizzles, otherwise it replaces both
    // the handles and shapes with the swizzled layer output handles and shapes
    if (!TransposeInputTensors(data, inputHandles, inputShapes, permutationPair.first))
    {
        return false;
    }

    // Create an armnn concat layer descriptor - this will also perform validation on the input shapes
    armnn::OriginsDescriptor concatDescriptor;

    try
    {
        // The concat descriptor is always created across the only supported concat dimension
        // which is 0, 1 or 3 for a 4-D tensor, or 0 or 2 for a 3-D tensor.
        concatDescriptor = armnn::CreateDescriptorForConcatenation(inputShapes.begin(),
                                                                   inputShapes.end(),
                                                                   concatDim);
    } catch (std::exception& error)
    {
        return Fail("%s: Error preparing concat descriptor. %s", __func__, error.what());
    }

    // Validate the output shape is correct given the input shapes based on the
    // only valid concat dimension which is 0, 1 or 3 for a 4-D tensor, or 0 or 2 for a 3-D tensor.
    if (!isDynamicTensor)
    {
        if (!ValidateConcatOutputShape(inputShapes, outputShape, concatDim))
        {
            return Fail("%s: Error validating the output shape for concat", __func__);
        }
    }

    std::vector<const armnn::TensorInfo*> inputTensorInfos;
    std::transform(inputHandles.begin(), inputHandles.end(), std::back_inserter(inputTensorInfos),
                   [](const LayerInputHandle& h)->const armnn::TensorInfo*{ return &h.GetTensorInfo(); });

    bool isSupported  = false;
    armnn::BackendId setBackendConcat;
    auto validateFunc = [&](const armnn::TensorInfo& outputInfo, bool& isSupported){
        FORWARD_LAYER_SUPPORT_FUNC(__func__,
                                   IsConcatSupported,
                                   data.m_Backends,
                                   isSupported,
                                   setBackendConcat,
                                   inputTensorInfos,
                                   outputInfo,
                                   concatDescriptor);
    };

    if (!isDynamicTensor)
    {
        validateFunc(outputInfo, isSupported);
    }
    else
    {
        isSupported = AreDynamicTensorsSupported();
    }

    if (!isSupported)
    {
        return false;
    }

    armnn::IConnectableLayer* layer = data.m_Network->AddConcatLayer(concatDescriptor);
    layer->SetBackendId(setBackendConcat);
    assert(layer != nullptr);
    layer->GetOutputSlot(0).SetTensorInfo(outputInfo);
    // Connect inputs to the layer
    const int numInputSlots = layer->GetNumInputSlots();
    assert(static_cast<std::size_t>(numInputSlots) == inputHandles.size());
    for (int i = 0; i < numInputSlots; ++i)
    {
        // connect the input directly to the merge (concat) layer
        inputHandles[static_cast<unsigned int>(i)].Connect(layer->GetInputSlot(i));
    }

    // Transpose the output shape
    auto transposeOutputShape = [&](){
        armnn::TransposeDescriptor transposeDesc;
        transposeDesc.m_DimMappings = permutationPair.second;
        armnn::TensorInfo inputTransposeInfo  = layer->GetOutputSlot(0).GetTensorInfo();
        armnn::TensorInfo outputTransposeInfo = armnnUtils::TransposeTensorShape(inputTransposeInfo,
                                                                                 permutationPair.second);
        isSupported = false;
        armnn::BackendId setBackendTranspose;
        FORWARD_LAYER_SUPPORT_FUNC(__func__,
                                   IsTransposeSupported,
                                   data.m_Backends,
                                   isSupported,
                                   setBackendTranspose,
                                   inputTransposeInfo,
                                   outputTransposeInfo,
                                   transposeDesc);
        if (!isSupported)
        {
            return false;
        }
        // Add permutation layer and connect the output to it, the permutation becomes the output layer
        armnn::IConnectableLayer& deswizzleLayer = AddTransposeLayer(*data.m_Network, layer->GetOutputSlot(0),
                                                                     permutationPair.second);
        deswizzleLayer.SetBackendId(setBackendTranspose);
        layer = &deswizzleLayer;

        return true;
    };

    if (needPermute && !isDynamicTensor)
    {
        transposeOutputShape();
    }

    if (inputsHaveBeenReshaped)
    {
        if (isDynamicTensor)
        {
            // Infer the output shapes of concat if outputs are type 1 dynamic
            ARMNN_ASSERT(layer->GetOutputSlot(0).IsTensorInfoSet());
            if (!ValidateConcatOutputShape(inputShapes,
                                           layer->GetOutputSlot(0).GetTensorInfo().GetShape(),
                                           concatDim))
            {
                return Fail("%s: Error validating the output shape for concat", __func__);
            }
            transposeOutputShape();
        }

        armnn::TensorInfo afterConcatInfo = layer->GetOutputSlot(0).GetTensorInfo();
        // Undo the reshape knowing the amount of dimensions added
        if (tensorDimensionsAdded == 1)
        {
            afterConcatInfo.SetShape(
                    armnn::TensorShape({afterConcatInfo.GetShape()[1], afterConcatInfo.GetShape()[2]}));
        }
        else if (tensorDimensionsAdded == 2)
        {
            afterConcatInfo.SetShape(armnn::TensorShape({afterConcatInfo.GetShape()[2]}));
        }

        armnn::ReshapeDescriptor reshapeDescriptor;
        reshapeDescriptor.m_TargetShape = afterConcatInfo.GetShape();
        armnn::TensorInfo concatInfo = layer->GetOutputSlot(0).GetTensorInfo();

        isSupported = false;
        armnn::BackendId setBackendReshape2;
        auto validateReshapeFunc = [&](const armnn::TensorInfo& afterConcatInfo, bool& isSupported){
            FORWARD_LAYER_SUPPORT_FUNC(__func__,
                                       IsReshapeSupported,
                                       data.m_Backends,
                                       isSupported,
                                       setBackendReshape2,
                                       concatInfo,
                                       afterConcatInfo,
                                       reshapeDescriptor);
        };

        if (!IsDynamicTensor(afterConcatInfo))
        {
            validateReshapeFunc(afterConcatInfo, isSupported);
        }
        else
        {
            isSupported = AreDynamicTensorsSupported();
        }

        if (!isSupported)
        {
            return false;
        }
        layer = &AddReshapeLayer(*data.m_Network, layer->GetOutputSlot(0), afterConcatInfo);
        layer->SetBackendId(setBackendReshape2);
        return SetupAndTrackLayerOutputSlot(operation,
                                            0,
                                            *layer,
                                            model,
                                            data,
                                            nullptr,
                                            validateReshapeFunc);
    }

    return SetupAndTrackLayerOutputSlot(operation, 0, *layer, model, data, nullptr, validateFunc);
}

bool Converter::ConvertConv2d(const Operation& operation, const Model& model, ConversionData& data)
{
    VLOG(DRIVER) << "Converter::ConvertConv2d()";

    LayerInputHandle input = ConvertToLayerInputHandle(operation, 0, model, data);
    if (!input.IsValid())
    {
        return Fail("%s: Operation has invalid inputs", __func__);
    }

    const Operand* output = GetOutputOperand(operation, 0, model);
    if (!output)
    {
        return Fail("%s: Could not read output 0", __func__);
    }

    const TensorInfo& inputInfo  = input.GetTensorInfo();
    const TensorInfo& outputInfo = GetTensorInfoForOperand(*output);

    Convolution2dDescriptor desc;
    desc.m_DataLayout = DataLayout::NHWC;

    // Determine whether padding is implicit or explicit
    bool implicitPadding = operation.inputs.size() == 7
                            || (operation.inputs.size() >= 8
                                 && GetInputOperand(operation, 7, model)->type == OperandType::BOOL);

    if (implicitPadding)
    {
        desc.m_DataLayout = OptionalDataLayout(operation, 7, model, data);
    }
    else if (operation.inputs.size() >= 10)
    {
        desc.m_DataLayout = OptionalDataLayout(operation, 10, model, data);
    }

    const PermutationVector OHWIToOIHW = {0, 2, 3, 1};

    // ArmNN does not currently support non-fixed weights or bias
    // The NNAPI filter is always OHWI [depth_out, filter_height, filter_width, depth_in] but ArmNN expects the
    // filter's height and width indices to match the input's height and width indices so we permute it to OIHW if
    // the DataLayout is NCHW

    if (!IsWeightsValid(operation, 1, model) && desc.m_DataLayout == DataLayout::NCHW)
    {
        return Fail("%s: Operation has unsupported weights OperandLifeTime", __func__);
    }

    LayerInputHandle weightsInput = (desc.m_DataLayout == DataLayout::NCHW)
                                      ? ConvertToLayerInputHandle(operation, 1, model, data, OHWIToOIHW, &input)
                                      : ConvertToLayerInputHandle(operation, 1, model, data, g_DontPermute, &input);

    if (!weightsInput.IsValid())
    {
        return Fail("%s: Operation has invalid inputs", __func__);
    }

    LayerInputHandle biasInput = ConvertToLayerInputHandle(operation, 2, model, data, g_DontPermute, &input); // 1D
    if (!biasInput.IsValid())
    {
        return Fail("%s: Operation has invalid inputs", __func__);
    }

    biasInput.SanitizeQuantizationScale(weightsInput, input);
    armnn::TensorInfo weightsInfo = weightsInput.GetTensorInfo();
    armnn::TensorInfo biasInfo    = biasInput.GetTensorInfo();

    ActivationFn activation;
    if (implicitPadding)
    {
        ::android::nn::PaddingScheme paddingScheme;
        if (!GetInputPaddingScheme(operation, 3, paddingScheme, model, data)
              || !GetInputScalar(operation, 4, OperandType::INT32, desc.m_StrideX, model, data)
              || !GetInputScalar(operation, 5, OperandType::INT32, desc.m_StrideY, model, data)
              || !GetInputActivationFunction(operation, 6, activation, model, data)
              || !GetOptionalConvolutionDilationParams(operation, 8, desc, model, data))
        {
            return Fail("%s: Operation has invalid inputs (implicit padding)", __func__);
        }

        armnnUtils::DataLayoutIndexed dataLayoutIndexed(desc.m_DataLayout);
        unsigned int widthIndex = dataLayoutIndexed.GetWidthIndex();
        unsigned int heightIndex = dataLayoutIndexed.GetHeightIndex();
        const uint32_t kernelX = weightsInfo.GetShape()[widthIndex];
        const uint32_t kernelY = weightsInfo.GetShape()[heightIndex];
        const uint32_t inputX  = inputInfo.GetShape()[widthIndex];
        const uint32_t inputY  = inputInfo.GetShape()[heightIndex];

        CalcPadding(inputX, kernelX, desc.m_StrideX, desc.m_DilationX, desc.m_PadLeft, desc.m_PadRight, paddingScheme);
        CalcPadding(inputY, kernelY, desc.m_StrideY, desc.m_DilationY, desc.m_PadTop, desc.m_PadBottom, paddingScheme);

    }
    else if (operation.inputs.size() >= 10)
    {
        // explicit padding
        if (!GetInputScalar(operation, 3, OperandType::INT32, desc.m_PadLeft, model, data)
              || !GetInputScalar(operation, 4, OperandType::INT32, desc.m_PadRight, model, data)
              || !GetInputScalar(operation, 5, OperandType::INT32, desc.m_PadTop, model, data)
              || !GetInputScalar(operation, 6, OperandType::INT32, desc.m_PadBottom, model, data)
              || !GetInputScalar(operation, 7, OperandType::INT32, desc.m_StrideX, model, data)
              || !GetInputScalar(operation, 8, OperandType::INT32, desc.m_StrideY, model, data)
              || !GetInputActivationFunction(operation, 9, activation, model, data)
              || !GetOptionalConvolutionDilationParams(operation, 11, desc, model, data))
        {
            return Fail("%s: Operation has invalid inputs (explicit padding)", __func__);
        }
    }
    else
    {
        return Fail("%s: Unsupported number of operation inputs", __func__);
    }

    desc.m_BiasEnabled = true;
    Optional<TensorInfo> biases(biasInfo);

    bool requiresValidation = true;
    const Operand* weightsOperand = GetInputOperand(operation, 1, model);
    const Operand* biasOperand = GetInputOperand(operation, 2, model);
    if (IsConnectedToDequantize(weightsInput.GetOutputSlot())
                || IsConnectedToDequantize(biasInput.GetOutputSlot()))
    {
        // Do not require validation for now. There will be an optimization step
        // [ConvertConstDequantisationLayersToConstLayers] will convert layers to Constant layers
        // then at the end of the optimization there will be layer supported validation.
        requiresValidation = false;
        VLOG(DRIVER) << "Converter::ConvertConv2d(): Weights and Biases are as INPUTS.";
    }

    armnn::BackendId setBackend;
    auto validateFunc = [&](const armnn::TensorInfo& outputInfo, bool& isSupported) {
        FORWARD_LAYER_SUPPORT_FUNC(__func__,
                                   IsConvolution2dSupported,
                                   data.m_Backends,
                                   isSupported,
                                   setBackend,
                                   inputInfo,
                                   outputInfo,
                                   desc,
                                   weightsInfo,
                                   biases);
    };

    if (requiresValidation)
    {
        VLOG(DRIVER) << "Converter::ConvertConv2d(): Requires Validation!";
        bool isSupported = false;
        if (!IsDynamicTensor(outputInfo))
        {
            validateFunc(outputInfo, isSupported);
        }
        else
        {
            isSupported = AreDynamicTensorsSupported();
        }

        if (!isSupported)
        {
            return false;
        }
    }

    armnn::IConnectableLayer* startLayer = data.m_Network->AddConvolution2dLayer(desc);
    startLayer->SetBackendId(setBackend);

    if (!startLayer)
    {
        return Fail("%s: AddConvolution2dLayer failed", __func__);
    }

    input.Connect(startLayer->GetInputSlot(0));
    weightsInput.Connect(startLayer->GetInputSlot(1));
    biasInput.Connect(startLayer->GetInputSlot(2));

    return SetupAndTrackLayerOutputSlot(operation, 0, *startLayer, model, data, nullptr, validateFunc, activation);
}

bool Converter::ConvertDepthToSpace(const Operation& operation, const Model& model, ConversionData& data)
{
    VLOG(DRIVER) << "Converter::ConvertDepthToSpace()";

    LayerInputHandle input = ConvertToLayerInputHandle(operation, 0, model, data);
    if (!input.IsValid() )
    {
        return Fail("%s: Operation has invalid inputs", __func__);
    }

    const armnn::TensorInfo& inputInfo = input.GetTensorInfo();
    unsigned int rank = inputInfo.GetNumDimensions();
    if (rank != 4)
    {
        return Fail("%s: Only inputs with rank 4 are supported", __func__);
    }

    const Operand* output = GetOutputOperand(operation, 0, model);
    if (!output)
    {
        return Fail("%s: Could not read output 0", __func__);
    }

    const armnn::TensorInfo& outputInfo = GetTensorInfoForOperand(*output);

    armnn::DepthToSpaceDescriptor descriptor;

    GetInputScalar(operation, 1, OperandType::INT32, descriptor.m_BlockSize, model, data);
    if (descriptor.m_BlockSize <= 1)
    {
        return Fail("%s: Block size must be at least 1 in all dimensions");
    }

    descriptor.m_DataLayout = armnn::DataLayout::NHWC;
    if (Is12OrLaterOperand(*output))
    {
        descriptor.m_DataLayout = OptionalDataLayout(operation, 2, model, data);
    }

    bool isSupported = false;
    armnn::BackendId setBackend;
    auto validateFunc = [&](const armnn::TensorInfo& outputInfo, bool& isSupported)
    {
        FORWARD_LAYER_SUPPORT_FUNC(__func__,
                                   IsDepthToSpaceSupported,
                                   data.m_Backends,
                                   isSupported,
                                   setBackend,
                                   inputInfo,
                                   outputInfo,
                                   descriptor);
    };

    if(!IsDynamicTensor(outputInfo))
    {
        validateFunc(outputInfo, isSupported);
    }
    else
    {
        isSupported = AreDynamicTensorsSupported();
    }

    if (!isSupported)
    {
        return false;
    }

    armnn::IConnectableLayer* const layer = data.m_Network->AddDepthToSpaceLayer(descriptor);
    layer->SetBackendId(setBackend);
    assert(layer != nullptr);
    input.Connect(layer->GetInputSlot(0));

    return SetupAndTrackLayerOutputSlot(operation, 0, *layer, model, data, nullptr, validateFunc);
}

bool Converter::ConvertDepthwiseConv2d(const Operation& operation, const Model& model, ConversionData& data)
{
    VLOG(DRIVER) << "Converter::ConvertDepthwiseConv2d()";

    LayerInputHandle input = ConvertToLayerInputHandle(operation, 0, model, data);

    if (!input.IsValid())
    {
        return Fail("%s: Operation has invalid inputs", __func__);
    }

    const Operand* output = GetOutputOperand(operation, 0, model);

    if (!output)
    {
        return Fail("%s: Could not read output 0", __func__);
    }

    const armnn::TensorInfo& inputInfo  = input.GetTensorInfo();
    const armnn::TensorInfo& outputInfo = GetTensorInfoForOperand(*output);

    // ArmNN does not currently support non-fixed weights or bias
    // Find the shape of the weights tensor. In AndroidNN this will be [ 1, H, W, I * M ]
    const Operand* weightsOperand = GetInputOperand(operation, 1, model);

    if (!weightsOperand)
    {
        return Fail("%s: Could not read weights", __func__);
    }
    // Basic sanity check on the weights shape.
    // ANEURALNETWORKS_DEPTHWISE_CONV_2D specifies a 4-D tensor, of shape
    // [1, filter_height, filter_width, depth_out]
    if (weightsOperand->dimensions[0] != 1)
    {
        return Fail("%s: Filter operand dimension 0 is invalid, should be 1", __func__);
    }

    armnn::DepthwiseConvolution2dDescriptor desc;
    desc.m_DataLayout = armnn::DataLayout::NHWC;

    // Determine whether padding is implicit or explicit
    bool implicitPadding = operation.inputs.size() == 8
                           || (operation.inputs.size() >= 9
                           && GetInputOperand(operation, 8, model)->type == OperandType::BOOL);

    // Look ahead to find the optional DataLayout, if present
    const uint32_t dataLayoutFlagIndex = implicitPadding ? 8 : 11;
    desc.m_DataLayout = OptionalDataLayout(operation, dataLayoutFlagIndex, model, data);

    armnnUtils::DataLayoutIndexed dataLayoutIndexed(desc.m_DataLayout);
    unsigned int widthIndex  = dataLayoutIndexed.GetWidthIndex();
    unsigned int heightIndex = dataLayoutIndexed.GetHeightIndex();

    LayerInputHandle weightsInput = ConvertToLayerInputHandle(operation, 1, model, data, g_DontPermute, &input);
    if (!weightsInput.IsValid())
    {
        return Fail("%s: Operation has invalid inputs", __func__);
    }

    const Operand* biasOperand = GetInputOperand(operation, 2, model);
    if (!biasOperand)
    {
        return Fail("%s: Could not read bias", __func__);
    }

    LayerInputHandle biasInput = ConvertToLayerInputHandle(operation, 2, model, data, g_DontPermute, &input); // 1D
    if (!biasInput.IsValid())
    {
        return Fail("%s: Operation has invalid inputs", __func__);
    }

    biasInput.SanitizeQuantizationScale(weightsInput, input);
    armnn::TensorInfo weightsInfo = weightsInput.GetTensorInfo();
    armnn::TensorInfo biasInfo    = biasInput.GetTensorInfo();

    ActivationFn activation;
    if (implicitPadding)
    {
        ::android::nn::PaddingScheme paddingScheme;
        if (!GetInputPaddingScheme(operation, 3, paddingScheme, model, data)
                || !GetInputScalar(operation, 4, OperandType::INT32, desc.m_StrideX, model, data)
                || !GetInputScalar(operation, 5, OperandType::INT32, desc.m_StrideY, model, data)
                || !GetInputActivationFunction(operation, 7, activation, model, data)
                || !GetOptionalConvolutionDilationParams(operation, 9, desc, model, data))
        {
            return Fail("%s: Operation has invalid inputs (implicit padding)", __func__);
        }

        const uint32_t kernelX = weightsInfo.GetShape()[2];
        const uint32_t kernelY = weightsInfo.GetShape()[1];
        const uint32_t inputX  = inputInfo.GetShape()[widthIndex];
        const uint32_t inputY  = inputInfo.GetShape()[heightIndex];

        CalcPadding(inputX, kernelX, desc.m_StrideX, desc.m_DilationX, desc.m_PadLeft, desc.m_PadRight, paddingScheme);
        CalcPadding(inputY, kernelY, desc.m_StrideY, desc.m_DilationY, desc.m_PadTop, desc.m_PadBottom, paddingScheme);
    }
    else if (operation.inputs.size() >= 11)
    {
        // explicit padding
        if (!GetInputScalar(operation, 3, OperandType::INT32, desc.m_PadLeft, model, data)
                || !GetInputScalar(operation, 4, OperandType::INT32, desc.m_PadRight, model, data)
                || !GetInputScalar(operation, 5, OperandType::INT32, desc.m_PadTop, model, data)
                || !GetInputScalar(operation, 6, OperandType::INT32, desc.m_PadBottom, model, data)
                || !GetInputScalar(operation, 7, OperandType::INT32, desc.m_StrideX, model, data)
                || !GetInputScalar(operation, 8, OperandType::INT32, desc.m_StrideY, model, data)
                || !GetInputActivationFunction(operation,  10, activation, model, data)
                || !GetOptionalConvolutionDilationParams(operation, 12, desc, model, data))
        {
            return Fail("%s: Operation has invalid inputs (explicit padding)", __func__);
        }
    }
    else
    {
        return Fail("%s: Unsupported number of operation inputs", __func__);
    }

    desc.m_BiasEnabled = true;
    Optional<TensorInfo> biases(biasInfo);

    bool requiresValidation = true;
    if (IsConnectedToDequantize(weightsInput.GetOutputSlot()) || IsConnectedToDequantize(biasInput.GetOutputSlot()))
    {
        // Do not require validation for now. There will be an optimization step
        // [ConvertConstDequantisationLayersToConstLayers] will convert layers to Constant layers
        // then at the end of the optimization there will be layer supported validation.
        requiresValidation = false;
        VLOG(DRIVER) << "Converter::ConvertDepthwiseConv2d(): Weights and Biases are as INPUTS.";
    }

    armnn::BackendId setBackend;
    auto validateFunc = [&](const armnn::TensorInfo& outputInfo, bool& isSupported) {
        FORWARD_LAYER_SUPPORT_FUNC(__func__,
                                   IsDepthwiseConvolutionSupported,
                                   data.m_Backends,
                                   isSupported,
                                   setBackend,
                                   inputInfo,
                                   outputInfo,
                                   desc,
                                   weightsInfo,
                                   biases);
    };

    if (requiresValidation)
    {
        VLOG(DRIVER) << "Converter::ConvertDepthwiseConv2d(): Requires Validation!";
        bool isSupported = false;
        if (!IsDynamicTensor(outputInfo))
        {
            validateFunc(outputInfo, isSupported);
        }
        else
        {
            isSupported = AreDynamicTensorsSupported();
        }

        if (!isSupported)
        {
            return false;
        }
    }

    armnn::IConnectableLayer* startLayer = data.m_Network->AddDepthwiseConvolution2dLayer(desc);
    startLayer->SetBackendId(setBackend);

    if (!startLayer)
    {
        return Fail("%s: AddDepthwiseConvolution2dLayer failed", __func__);
    }

    input.Connect(startLayer->GetInputSlot(0));

    // Connect weights and bias inputs
    weightsInput.Connect(startLayer->GetInputSlot(1));
    biasInput.Connect(startLayer->GetInputSlot(2));

    return SetupAndTrackLayerOutputSlot(operation, 0, *startLayer, model, data, nullptr, validateFunc, activation);
}

bool Converter::ConvertDequantize(const Operation& operation, const Model& model, ConversionData& data)
{
    VLOG(DRIVER) << "Converter::ConvertDequantize()";

    LayerInputHandle input = ConvertToLayerInputHandle(operation, 0, model, data);
    if (!input.IsValid())
    {
        return Fail("%s: Operation has invalid input", __func__);
    }

    const armnn::TensorInfo& inputInfo  = input.GetTensorInfo();
    const armnn::Optional<unsigned int>& quantizationDim = inputInfo.GetQuantizationDim();
    if (quantizationDim.has_value() && quantizationDim.value() != 0)
    {
        return Fail("%s: Operation has quantization dimension different than 0", __func__);
    }

    const Operand* const outputOperand = GetOutputOperand(operation, 0, model);
    if (!outputOperand)
    {
        return Fail("%s: Operation has invalid outputs", __func__);
    }

    const armnn::TensorInfo& outputInfo = GetTensorInfoForOperand(*outputOperand);

    bool isSupported = false;
    armnn::BackendId setBackend;
    auto validateFunc = [&](const armnn::TensorInfo& outputInfo, bool& isSupported)
    {
        FORWARD_LAYER_SUPPORT_FUNC(__func__,
                                   IsDequantizeSupported,
                                   data.m_Backends,
                                   isSupported,
                                   setBackend,
                                   inputInfo,
                                   outputInfo);
    };

    if(IsDynamicTensor(outputInfo))
    {
        isSupported = AreDynamicTensorsSupported();
    }
    else
    {
        validateFunc(outputInfo, isSupported);
    }

    if (!isSupported)
    {
        return false;
    }

    armnn::IConnectableLayer* const layer = data.m_Network->AddDequantizeLayer();
    layer->SetBackendId(setBackend);
    assert(layer != nullptr);
    input.Connect(layer->GetInputSlot(0));

    return SetupAndTrackLayerOutputSlot(operation, 0, *layer, model, data, nullptr, validateFunc);
}

bool Converter::ConvertDiv(const Operation& operation, const Model& model, ConversionData& data)
{
    VLOG(DRIVER) << "Converter::ConvertDiv()";

    LayerInputHandle input0 = ConvertToLayerInputHandle(operation, 0, model, data);
    LayerInputHandle input1 = ConvertToLayerInputHandle(operation, 1, model, data);

    if (!input0.IsValid() || !input1.IsValid())
    {
        return Fail("%s: Operation has invalid inputs", __func__);
    }

    // The FuseActivation parameter is always the input index 2
    // and it should be optional
    ActivationFn activationFunction;
    if (!GetOptionalInputActivation(operation, 2, activationFunction, model, data))
    {
        return Fail("%s: Operation has invalid inputs", __func__);
    }

    const Operand* output = GetOutputOperand(operation, 0, model);
    if (!output)
    {
        return Fail("%s: Could not read output 0", __func__);
    }

    const armnn::TensorInfo& outputInfo = GetTensorInfoForOperand(*output);

    bool isSupported = false;
    armnn::BackendId setBackend;
    auto validateFunc = [&](const armnn::TensorInfo& outputInfo, bool& isSupported)
    {
        ARMNN_NO_DEPRECATE_WARN_BEGIN
        FORWARD_LAYER_SUPPORT_FUNC(__func__,
                                   IsDivisionSupported,
                                   data.m_Backends,
                                   isSupported,
                                   setBackend,
                                   input0.GetTensorInfo(),
                                   input1.GetTensorInfo(),
                                   outputInfo);
        ARMNN_NO_DEPRECATE_WARN_END
    };

    if(!IsDynamicTensor(outputInfo))
    {
        validateFunc(outputInfo, isSupported);
    }
    else
    {
        isSupported = AreDynamicTensorsSupported();
    }

    if (!isSupported)
    {
        return false;
    }

    ARMNN_NO_DEPRECATE_WARN_BEGIN
    armnn::IConnectableLayer* const startLayer = data.m_Network->AddDivisionLayer();
    ARMNN_NO_DEPRECATE_WARN_END
    startLayer->SetBackendId(setBackend);

    bool isReshapeSupported = BroadcastTensor(input0, input1, startLayer, data);
    if (!isReshapeSupported)
    {
        return false;
    }

    return SetupAndTrackLayerOutputSlot(operation, 0, *startLayer, model,
                                        data, nullptr, validateFunc, activationFunction);
}

bool Converter::ConvertElementwiseUnary(const Operation& operation,
                                        const Model& model,
                                        ConversionData& data,
                                        UnaryOperation unaryOperation)
{
    VLOG(DRIVER) << "Converter::ConvertElementwiseUnary()";
    VLOG(DRIVER) << "unaryOperation = " << GetUnaryOperationAsCString(unaryOperation);

    LayerInputHandle input = ConvertToLayerInputHandle(operation, 0, model, data);

    if (!input.IsValid())
    {
        return Fail("%s: Operation has invalid input", __func__);
    }

    const Operand* output = GetOutputOperand(operation, 0, model);
    if (!output)
    {
        return Fail("%s: Could not read output 0", __func__);
    }

    const TensorInfo& inputInfo = input.GetTensorInfo();
    const TensorInfo& outputInfo = GetTensorInfoForOperand(*output);

    ElementwiseUnaryDescriptor descriptor(unaryOperation);

    bool isSupported = false;
    armnn::BackendId setBackend;
    auto validateFunc = [&](const armnn::TensorInfo& outputInfo, bool& isSupported)
    {
        FORWARD_LAYER_SUPPORT_FUNC(__func__,
                                   IsElementwiseUnarySupported,
                                   data.m_Backends,
                                   isSupported,
                                   setBackend,
                                   inputInfo,
                                   outputInfo,
                                   descriptor);
    };

    if(!IsDynamicTensor(outputInfo))
    {
        validateFunc(outputInfo, isSupported);
    }
    else
    {
        isSupported = AreDynamicTensorsSupported();
    }

    if (!isSupported)
    {
        return false;
    }

    IConnectableLayer* layer = data.m_Network->AddElementwiseUnaryLayer(descriptor);
    layer->SetBackendId(setBackend);
    assert(layer != nullptr);
    input.Connect(layer->GetInputSlot(0));

    return SetupAndTrackLayerOutputSlot(operation, 0, *layer, model, data, nullptr, validateFunc);
}

bool Converter::ConvertElu(const Operation& operation, const Model& model, ConversionData& data)
{
    VLOG(DRIVER) << "Converter::ConvertElu()";

    LayerInputHandle input0 = ConvertToLayerInputHandle(operation, 0, model, data);
    if (!input0.IsValid())
    {
        return Fail("%s: Operation has invalid inputs", __func__);
    }

    // Determine data type of input tensor
    OperandType inputType;
    if (!GetOperandType(operation, 0, model, inputType))
    {
        return Fail("%s: Operation has invalid inputs", __func__);
    }

    ActivationDescriptor desc;
    desc.m_Function = ActivationFunction::Elu;

    // Read alpha
    if (inputType == OperandType::TENSOR_FLOAT16)
    {
        Half alpha;

        if (!GetInputScalar(operation, 1, OperandType::FLOAT16, alpha, model, data))
        {
            return Fail("%s: Operation has invalid inputs (FLOAT16)", __func__);
        }

        desc.m_A = static_cast<float>(alpha);
    }
    else if (inputType == OperandType::TENSOR_FLOAT32)
    {
        if (!GetInputScalar(operation, 1, OperandType::FLOAT32, desc.m_A, model, data))
        {
            return Fail("%s: Operation has invalid inputs (FLOAT32)", __func__);
        }
    }
    else
    {
        return Fail("%s: Unsupported input tensor type: %d", __func__, inputType);
    }

    return ::ConvertToActivation(operation, __func__, desc, model, data);
}

bool Converter::ConvertExpandDims(const Operation& operation, const Model& model, ConversionData& data)
{
    VLOG(DRIVER) << "Converter::ConvertExpandDims()";

    LayerInputHandle input = ConvertToLayerInputHandle(operation, 0, model, data);

    if (!input.IsValid())
    {
        return Fail("%s: Operation has invalid input", __func__);
    }

    const Operand* output = GetOutputOperand(operation, 0, model);
    if (!output)
    {
        return Fail("%s: Operation has invalid output", __func__);
    }

    const TensorInfo& outputInfo = GetTensorInfoForOperand(*output);

    int32_t axis;
    if (!GetInputScalar(operation, 1, OperandType::INT32, axis, model, data))
    {
        return Fail("%s: failed to get axis input value", __func__);
    }

    TensorShape targetShape;

    try
    {
        targetShape = armnnUtils::ExpandDims(input.GetTensorInfo().GetShape(), axis);
    }
    catch (const std::exception& e)
    {
        return Fail("%s: %s", __func__, e.what());
    }

    ReshapeDescriptor reshapeDescriptor;
    reshapeDescriptor.m_TargetShape = targetShape;

    bool isSupported = false;
    armnn::BackendId setBackend;
    auto validateFunc = [&](const armnn::TensorInfo& outputInfo, bool& isSupported)
    {
        FORWARD_LAYER_SUPPORT_FUNC(__func__,
                                   IsReshapeSupported,
                                   data.m_Backends,
                                   isSupported,
                                   setBackend,
                                   input.GetTensorInfo(),
                                   outputInfo,
                                   reshapeDescriptor);
    };

    if(!IsDynamicTensor(outputInfo))
    {
        if (targetShape != outputInfo.GetShape())
        {
            return Fail("%s: Shape of the output operand does not match the resolved expanded shape", __func__);
        }
        validateFunc(outputInfo, isSupported);
    }
    else
    {
        isSupported = AreDynamicTensorsSupported();
    }

    if (!isSupported)
    {
        return false;
    }

    IConnectableLayer* layer = data.m_Network->AddReshapeLayer(reshapeDescriptor);
    layer->SetBackendId(setBackend);
    assert(layer != nullptr);
    input.Connect(layer->GetInputSlot(0));

    return SetupAndTrackLayerOutputSlot(operation, 0, *layer, model, data, nullptr, validateFunc);
}

bool Converter::ConvertFill(const Operation& operation, const Model& model, ConversionData& data)
{
    VLOG(DRIVER) << "Converter::ConvertFill()";
    LayerInputHandle input = ConvertToLayerInputHandle(operation, 0, model, data);
    if (!input.IsValid())
    {
        return Fail("%s: Operation has invalid inputs", __func__);
    }

    const Operand* output = GetOutputOperand(operation, 0, model);
    if (!output)
    {
        return Fail("%s: Could not read output", __func__);
    }

    const TensorInfo& inputInfo  = input.GetTensorInfo();
    const TensorInfo& outputInfo = GetTensorInfoForOperand(*output);
    if (IsDynamicTensor(outputInfo))
    {
        return Fail("%s: Dynamic output tensors are not supported", __func__);
    }

    // Determine data type of output tensor
    OperandType outputType = output->type;
    FillDescriptor descriptor;
    // Read the scalar fill value
    if (outputType == OperandType::TENSOR_FLOAT16)
    {
        Half value;

        if (!GetInputScalar(operation, 1, OperandType::FLOAT16, value, model, data))
        {
            return Fail("%s: Operation has invalid inputs %d", __func__, outputType);
        }

        descriptor.m_Value = static_cast<float>(value);
    }
    else if (outputType == OperandType::TENSOR_FLOAT32)
    {
        if (!GetInputScalar(operation, 1, OperandType::FLOAT32, descriptor.m_Value, model, data))
        {
            return Fail("%s: Operation has invalid inputs %d", __func__, outputType);
        }
    }
    else if (outputType == OperandType::TENSOR_INT32)
    {
        int32_t value;

        if (!GetInputScalar(operation, 1, OperandType::INT32, value, model, data))
        {
            return Fail("%s: Operation has invalid inputs %d", __func__, outputType);
        }

        descriptor.m_Value = static_cast<float>(value);
    }
    else
    {
        return Fail("%s: Unsupported input tensor type: %d", __func__, outputType);
    }

    bool isSupported = false;
    armnn::BackendId setBackend;
    FORWARD_LAYER_SUPPORT_FUNC(__func__,
                               IsFillSupported,
                               data.m_Backends,
                               isSupported,
                               setBackend,
                               inputInfo,
                               outputInfo,
                               descriptor);
    if (!isSupported)
    {
        return false;
    }

    IConnectableLayer* const layer = data.m_Network->AddFillLayer(descriptor);
    layer->SetBackendId(setBackend);
    assert(layer != nullptr);
    input.Connect(layer->GetInputSlot(0));

    return SetupAndTrackLayerOutputSlot(operation, 0, *layer, model, data);
}

bool Converter::ConvertFloor(const Operation& operation, const Model& model, ConversionData& data)
{
    VLOG(DRIVER) << "Converter::ConvertFloor()";
    LayerInputHandle input = ConvertToLayerInputHandle(operation, 0, model, data);
    if (!input.IsValid())
    {
        return Fail("%s: Operation has invalid inputs", __func__);
    }

    const Operand* const outputOperand = GetOutputOperand(operation, 0, model);
    if (!outputOperand)
    {
        return Fail("%s: Operation has invalid outputs", __func__);
    }

    const armnn::TensorInfo& outputInfo = GetTensorInfoForOperand(*outputOperand);

    bool isSupported = false;
    armnn::BackendId setBackend;
    auto validateFunc = [&](const armnn::TensorInfo& outputInfo, bool& isSupported)
    {
        FORWARD_LAYER_SUPPORT_FUNC(__func__,
                                   IsFloorSupported,
                                   data.m_Backends,
                                   isSupported,
                                   setBackend,
                                   input.GetTensorInfo(),
                                   outputInfo);
    };

    if(!IsDynamicTensor(outputInfo))
    {
        validateFunc(outputInfo, isSupported);
    }
    else
    {
        isSupported = AreDynamicTensorsSupported();
    }

    if (!isSupported)
    {
        return false;
    }

    armnn::IConnectableLayer* layer = data.m_Network->AddFloorLayer();
    layer->SetBackendId(setBackend);
    assert(layer != nullptr);
    input.Connect(layer->GetInputSlot(0));

    return SetupAndTrackLayerOutputSlot(operation, 0, *layer, model, data, nullptr, validateFunc);
}

bool Converter::ConvertFullyConnected(const Operation& operation, const Model& model, ConversionData& data)
{
    VLOG(DRIVER) << "Converter::ConvertFullyConnected()";
    LayerInputHandle input = ConvertToLayerInputHandle(operation, 0, model, data);
    if (!input.IsValid())
    {
        return Fail("%s: Operation has invalid inputs", __func__);
    }

    const Operand* output = GetOutputOperand(operation, 0, model);
    if (!output)
    {
        return Fail("%s: Could not read output 0", __func__);
    }

    const armnn::TensorInfo& inputInfo  = input.GetTensorInfo();
    const armnn::TensorInfo& outputInfo = GetTensorInfoForOperand(*output);

    LayerInputHandle weightsInput = LayerInputHandle();
    const Operand* weightsOperand = GetInputOperand(operation, 1, model);
    if (!weightsOperand)
    {
        return Fail("%s: Could not read weights", __func__);
    }

    // If weights are constant a separate constant layer will be created to store data.
    // Otherwise handle non const weights as inputs.
    weightsInput = ConvertToLayerInputHandle(operation, 1, model, data);
    if (!weightsInput.IsValid())
    {
        return Fail("%s: Operation has invalid inputs", __func__);
    }

    LayerInputHandle biasInput = LayerInputHandle();
    const Operand* biasOperand = GetInputOperand(operation, 2, model);
    if (!biasOperand)
    {
        return Fail("%s: Could not read bias", __func__);
    }

    // If bias are constant a separate constant layer will be created to store data.
    // Otherwise handle non const bias as inputs.
    biasInput = ConvertToLayerInputHandle(operation, 2, model, data); // 1D
    if (!biasInput.IsValid())
    {
        return Fail("%s: Operation has invalid inputs", __func__);
    }

    armnn::TensorInfo weightsInfo = weightsInput.GetTensorInfo();
    armnn::TensorInfo reshapedInfo = inputInfo;
    try
    {
        reshapedInfo.SetShape(FlattenFullyConnectedInput(inputInfo.GetShape(), weightsInfo.GetShape()));
    }
    catch (const std::exception& e)
    {
        return Fail("%s: %s", __func__, e.what());
    }

    // Ensuring that the bias value is within 1% of the weights input (small float differences can exist)
    armnn::TensorInfo biasInfo = biasInput.GetTensorInfo();
    SanitizeBiasQuantizationScale(biasInfo, weightsInfo, reshapedInfo);

    ActivationFn activationFunction;
    if (!GetInputActivationFunction(operation, 3, activationFunction, model, data))
    {
        return Fail("%s: Operation has invalid inputs", __func__);
    }

    armnn::FullyConnectedDescriptor desc;
    desc.m_TransposeWeightMatrix = true;
    desc.m_BiasEnabled           = true;
    desc.m_ConstantWeights       = IsOperandConstant(*weightsOperand);

    bool isSupported = false;
    armnn::BackendId setBackend;
    auto validateFunc = [&](const armnn::TensorInfo& outputInfo, bool& isSupported)
    {
        if (!VerifyFullyConnectedShapes(reshapedInfo.GetShape(),
                                        weightsInfo.GetShape(),
                                        outputInfo.GetShape(),
                                        desc.m_TransposeWeightMatrix))
        {
            isSupported = false;
            Fail("%s: Expected outputShape does not match actual outputShape", __func__);
            return;
        }

        FORWARD_LAYER_SUPPORT_FUNC(__func__,
                                   IsFullyConnectedSupported,
                                   data.m_Backends,
                                   isSupported,
                                   setBackend,
                                   reshapedInfo,
                                   outputInfo,
                                   weightsInfo,
                                   biasInfo,
                                   desc);
    };

    if(!IsDynamicTensor(outputInfo))
    {
        validateFunc(outputInfo, isSupported);
    }
    else
    {
        isSupported = AreDynamicTensorsSupported();
    }

    if (!isSupported)
    {
        return false;
    }

    // Add FullyConnected layer. Weights and bias will be connected as constant layers or non const inputs.
    armnn::IConnectableLayer* startLayer = data.m_Network->AddFullyConnectedLayer(desc);
    startLayer->SetBackendId(setBackend);

    if (inputInfo.GetNumDimensions() > 2U)
    {
        armnn::ReshapeDescriptor reshapeDescriptor;
        reshapeDescriptor.m_TargetShape = reshapedInfo.GetShape();

        armnn::IConnectableLayer* reshapeLayer = data.m_Network->AddReshapeLayer(reshapeDescriptor);
        assert(reshapeLayer != nullptr);
        input.Connect(reshapeLayer->GetInputSlot(0));
        reshapeLayer->GetOutputSlot(0).SetTensorInfo(reshapedInfo);
        reshapeLayer->GetOutputSlot(0).Connect(startLayer->GetInputSlot(0));
    }
    else
    {
        input.Connect(startLayer->GetInputSlot(0));
    }

    // Connect weights and bias inputs
    weightsInput.Connect(startLayer->GetInputSlot(1));
    biasInput.Connect(startLayer->GetInputSlot(2));

    return SetupAndTrackLayerOutputSlot(operation, 0, *startLayer, model,
                                        data, nullptr, validateFunc, activationFunction);
}

bool Converter::ConvertGather(const Operation& operation, const Model& model, ConversionData& data)
{
    VLOG(DRIVER) << "Converter::ConvertGather()";

    LayerInputHandle input = ConvertToLayerInputHandle(operation, 0, model, data);
    if (!input.IsValid())
    {
        return Fail("%s: Operation has invalid input", __func__);
    }
    auto inputDimensions = input.GetTensorInfo().GetNumDimensions();

    LayerInputHandle indices = ConvertToLayerInputHandle(operation, 2, model, data);
    if (!indices.IsValid())
    {
        return Fail("%s: Operation has invalid indices", __func__);
    }
    auto indicesDimensions = indices.GetTensorInfo().GetNumDimensions();

    const Operand* output = GetOutputOperand(operation, 0, model);
    if (!output)
    {
        return Fail("%s: Operation has invalid output", __func__);
    }
    const TensorInfo& outputInfo = GetTensorInfoForOperand(*output);
    auto outputDimensions = outputInfo.GetNumDimensions();
    if (outputDimensions != inputDimensions + indicesDimensions - 1)
    {
        return Fail("%s: Operation has invalid output dimensions: %d. Output must be an (%d + %d - 1)-D tensor",
                     __func__, outputDimensions, inputDimensions, indicesDimensions);
    }

    int32_t axis;
    if (!GetInputScalar(operation, 1, OperandType::INT32, axis, model, data))
    {
        return Fail("%s: Operation has invalid or unsupported axis operand", __func__);
    }
    if (((axis < -inputDimensions) && (axis < 0)) || ((axis >= inputDimensions) && (axis > 0)))
    {
        return Fail("%s: Operation has invalid axis: %d. It is out of bounds [-%d, %d))", __func__, axis,
                    inputDimensions, inputDimensions);
    }

    GatherDescriptor desc;
    desc.m_Axis = axis;

    bool isSupported = false;
    armnn::BackendId setBackend;
    auto validateFunc = [&](const armnn::TensorInfo& outputInfo, bool& isSupported)
    {
        FORWARD_LAYER_SUPPORT_FUNC(__func__,
                                   IsGatherSupported,
                                   data.m_Backends,
                                   isSupported,
                                   setBackend,
                                   input.GetTensorInfo(),
                                   indices.GetTensorInfo(),
                                   outputInfo,
                                   desc);
    };

    if(!IsDynamicTensor(outputInfo))
    {
        validateFunc(outputInfo, isSupported);
    }
    else
    {
        isSupported = AreDynamicTensorsSupported();
    }

    if (!isSupported)
    {
        return false;
    }

    IConnectableLayer* layer = data.m_Network->AddGatherLayer(desc);
    layer->SetBackendId(setBackend);
    assert(layer != nullptr);
    input.Connect(layer->GetInputSlot(0));
    indices.Connect(layer->GetInputSlot(1));

    return SetupAndTrackLayerOutputSlot(operation, 0, *layer, model, data, nullptr, validateFunc);
}

bool Converter::ConvertGroupedConv2d(const Operation& operation, const Model& model, ConversionData& data)
{
    VLOG(DRIVER) << "Converter::ConvertGroupedConv2d()";
    //
    // Parse data
    //
    LayerInputHandle input = ConvertToLayerInputHandle(operation, 0, model, data);
    if (!input.IsValid())
    {
        return Fail("%s: Operation has invalid inputs", __func__);
    }
    const TensorInfo& inputInfo  = input.GetTensorInfo();

    const Operand* output = GetOutputOperand(operation, 0, model);
    if (!output)
    {
        return Fail("%s: Could not read output 0", __func__);
    }
    TensorInfo outputInfo = GetTensorInfoForOperand(*output);

    // Look ahead to determine data layout
    DataLayout dataLayout = DataLayout::NHWC;
    if (operation.inputs.size() == 12)
    {
        dataLayout = OptionalDataLayout(operation, 11, model, data);
    }
    else
    {
        dataLayout = OptionalDataLayout(operation, 8, model, data);
    }

    // NOTE:
    // NNAPI weights are always OHWI, i.e. [depth_out, filter_height, filter_width, depth_group],
    // but Arm NN expects the filter's height and width indices to match the input's height and
    // width indices so when the DataLayout is NCHW, we need to permute the weights to OIHW
    const PermutationVector ohwiToOihw = { 0u, 2u, 3u, 1u };
    const ConstTensorPin weightsPin = (dataLayout == DataLayout::NCHW) ?
                                      ConvertOperationInputToConstTensorPin(operation, 1,
                                                                                       model, data, ohwiToOihw) :
                                      ConvertOperationInputToConstTensorPin(operation, 1, model, data);
    const ConstTensorPin biasesPin  =
        ConvertOperationInputToConstTensorPin(operation, 2, model, data);
    if (!weightsPin.IsValid() || !biasesPin.IsValid())
    {
        return Fail("%s: Operation has invalid inputs", __func__);
    }

    ConstTensor weights = weightsPin.GetConstTensor();
    ConstTensor biases  = biasesPin.GetConstTensor();
    SanitizeBiasQuantizationScale(biases.GetInfo(), weights.GetInfo(), inputInfo);

    const TensorShape& inputShape   = inputInfo.GetShape();
    const TensorShape& outputShape  = outputInfo.GetShape();
    const TensorShape& weightsShape = weights.GetShape();
    const TensorShape& biasesShape  = biases.GetShape();

    armnnUtils::DataLayoutIndexed dataLayoutIndexed(dataLayout);
    const unsigned int channelsIndex = dataLayoutIndexed.GetChannelsIndex();
    const unsigned int heightIndex   = dataLayoutIndexed.GetHeightIndex();
    const unsigned int widthIndex    = dataLayoutIndexed.GetWidthIndex();

    Convolution2dDescriptor desc;
    desc.m_DataLayout  = dataLayout;
    desc.m_BiasEnabled = true;

    int numGroups;
    ActivationFn activation;

    if (operation.inputs.size() == 12)
    {
        if (!GetInputScalar(operation, 3, OperandType::INT32, desc.m_PadLeft, model, data) ||
            !GetInputScalar(operation, 4, OperandType::INT32, desc.m_PadRight, model, data) ||
            !GetInputScalar(operation, 5, OperandType::INT32, desc.m_PadTop, model, data) ||
            !GetInputScalar(operation, 6, OperandType::INT32, desc.m_PadBottom, model, data) ||
            !GetInputScalar(operation, 7, OperandType::INT32, desc.m_StrideX, model, data) ||
            !GetInputScalar(operation, 8, OperandType::INT32, desc.m_StrideY, model, data) ||
            !GetInputScalar(operation, 9, OperandType::INT32, numGroups, model, data) ||
            !GetInputActivationFunction(operation, 10, activation, model, data))
        {
            return Fail("%s: Operation has invalid inputs (explicit padding)", __func__);
        }

    }
    else if (operation.inputs.size() == 9)
    {
        ::android::nn::PaddingScheme paddingScheme;
        if (!GetInputPaddingScheme(operation, 3, paddingScheme, model, data) ||
            !GetInputScalar(operation, 4, OperandType::INT32, desc.m_StrideX, model, data) ||
            !GetInputScalar(operation, 5, OperandType::INT32, desc.m_StrideY, model, data) ||
            !GetInputScalar(operation, 6, OperandType::INT32, numGroups, model, data) ||
            !GetInputActivationFunction(operation, 7, activation, model, data))
        {
            return Fail("%s: Operation has invalid inputs (implicit padding)", __func__);
        }

        const uint32_t inputX = inputInfo.GetShape()[widthIndex];
        const uint32_t inputY = inputInfo.GetShape()[heightIndex];

        const uint32_t kernelX = weightsShape[widthIndex];
        const uint32_t kernelY = weightsShape[heightIndex];

        CalcPadding(inputX, kernelX, desc.m_StrideX, desc.m_PadLeft, desc.m_PadRight, paddingScheme);
        CalcPadding(inputY, kernelY, desc.m_StrideY, desc.m_PadTop, desc.m_PadBottom, paddingScheme);
    }
    else
    {
        return Fail("%s: Unsupported number of operation inputs", __func__);
    }

    // Equivalent to outputShape[channelsIndex], but we can't know the outputShape in the case of dynamic tensors
    const unsigned int outputChannels = weightsShape[0];

    const unsigned int channelsPerGroup  = weightsShape[channelsIndex];
    const unsigned int channelMultiplier = outputChannels / numGroups;

    //
    // Validate all relevant inputs
    //
    if (numGroups <= 0)
    {
        return Fail("%s: Number of groups must be greater than 0. Got: %d", __func__, numGroups);
    }

    if (outputChannels % numGroups != 0u)
    {
        return Fail("%s: Output channels must be divisible by the number of groups", __func__);
    }

    //
    // Set up Splitter layer
    //
    unsigned int splitterDimSizes[4] = { inputShape[0], inputShape[1], inputShape[2], inputShape[3] };
    splitterDimSizes[channelsIndex] /= numGroups; // split in depth

    TensorInfo splitterOutputInfo(4,
                                  splitterDimSizes,
                                  inputInfo.GetDataType(),
                                  inputInfo.GetQuantizationScale(),
                                  inputInfo.GetQuantizationOffset());

    std::vector<std::reference_wrapper<TensorInfo>> splitterOutputInfos(numGroups, std::ref(splitterOutputInfo));

    ViewsDescriptor splitterDesc(numGroups);
    for (unsigned int group = 0u; group < numGroups; ++group)
    {
        splitterDesc.SetViewOriginCoord(group, channelsIndex, splitterDimSizes[channelsIndex] * group);
        for (unsigned int dimIdx = 0u; dimIdx < 4u; dimIdx++)
        {
            splitterDesc.SetViewSize(group, dimIdx, splitterDimSizes[dimIdx]);
        }
    }

    bool isSupported = false;
    armnn::BackendId setBackendSplit;
    FORWARD_LAYER_SUPPORT_FUNC(__func__,
                               IsSplitterSupported,
                               data.m_Backends,
                               isSupported,
                               setBackendSplit,
                               inputInfo,
                               splitterOutputInfos,
                               splitterDesc);
    if (!isSupported)
    {
        return false;
    }

    IConnectableLayer* splitterLayer = data.m_Network->AddSplitterLayer(splitterDesc);
    splitterLayer->SetBackendId(setBackendSplit);
    if (!splitterLayer)
    {
        return Fail("%s: Failed to add SplitterLayer", __func__);
    }

    input.Connect(splitterLayer->GetInputSlot(0));
    for (unsigned int group = 0u; group < splitterLayer->GetNumOutputSlots(); ++group)
    {
        splitterLayer->GetOutputSlot(group).SetTensorInfo(splitterOutputInfo);
    }

    //
    // Set up Convolution2d layers for each group
    //

    // Set up group tensor shapes
    TensorShape groupInputShape(inputShape);
    groupInputShape[channelsIndex] = channelsPerGroup;

    TensorShape groupWeightsShape(weightsShape);
    groupWeightsShape[0] /= channelMultiplier * numGroups;

    TensorShape groupBiasesShape({ 1 });

    // Set up group tensor infos
    TensorInfo groupInputInfo(inputInfo);
    groupInputInfo.SetShape(groupInputShape);

    const TensorInfo& weightsInfo = weights.GetInfo();
    TensorInfo groupWeightsInfo(weightsInfo);
    groupWeightsInfo.SetShape(groupWeightsShape);

    const TensorInfo& biasesInfo = biases.GetInfo();
    TensorInfo groupBiasesInfo(biasesInfo);
    groupBiasesInfo.SetShape(groupBiasesShape);

    TensorInfo groupOutputInfo(outputInfo);

    TensorShape groupOutputShape(outputShape);
    const bool isDynamic = IsDynamicTensor(outputInfo);
    if (!isDynamic)
    {
        groupOutputShape[channelsIndex] = 1;
    }
    groupOutputInfo.SetShape(groupOutputShape);

    const unsigned int weightsDataTypeSize = GetDataTypeSize(groupWeightsInfo.GetDataType());
    const unsigned int biasesDataTypeSize  = GetDataTypeSize(groupBiasesInfo.GetDataType());

    std::vector<IConnectableLayer*> convLayers(numGroups * channelMultiplier, nullptr);
    for (unsigned int group = 0u; group < numGroups; ++group)
    {
        for (unsigned int m = 0u; m < channelMultiplier; ++m)
        {
            auto index = group * channelMultiplier + m;

            const unsigned int weightsDataOffset = groupWeightsShape.GetNumElements() * index * weightsDataTypeSize;
            const unsigned int biasesDataOffset = groupBiasesShape.GetNumElements() * index * biasesDataTypeSize;

            if (weightsInfo.HasPerAxisQuantization())
            {
                // Extract per-axis quantization scales for group weights
                const std::vector<float>& weightsQuantScales = weightsInfo.GetQuantizationScales();
                groupWeightsInfo.SetQuantizationScales(
                    std::vector<float>(weightsQuantScales.begin() + index,
                                       weightsQuantScales.begin() + index + groupWeightsShape[0]));

                // Extract per-axis quantization scales for group biases
                const std::vector<float>& biasesQuantScales  = biasesInfo.GetQuantizationScales();
                groupBiasesInfo.SetQuantizationScales(
                    std::vector<float>(biasesQuantScales.begin() + index,
                                       biasesQuantScales.begin() + index + groupWeightsShape[0]));
            }

            // Extract weights and biases data for current group convolution
            ConstTensor groupWeights(groupWeightsInfo,
                                     static_cast<const void *>(reinterpret_cast<const char *>(weights.GetMemoryArea()) +
                                                               weightsDataOffset));
            ConstTensor groupBiases(groupBiasesInfo,
                                    static_cast<const void *>(reinterpret_cast<const char *>(biases.GetMemoryArea()) +
                                                              biasesDataOffset));

            isSupported = false;
            armnn::BackendId setBackendConv;
            auto validateFunc = [&](const armnn::TensorInfo& outputInfo, bool& isSupported)
            {
                FORWARD_LAYER_SUPPORT_FUNC(__func__,
                                           IsConvolution2dSupported,
                                           data.m_Backends,
                                           isSupported,
                                           setBackendConv,
                                           groupInputInfo,
                                           outputInfo,
                                           desc,
                                           groupWeightsInfo,
                                           Optional<TensorInfo>(groupBiasesInfo));
            };

            if(!isDynamic)
            {
                validateFunc(groupOutputInfo, isSupported);
            }
            else
            {
                isSupported = AreDynamicTensorsSupported();
            }

            if (!isSupported)
            {
                return false;
            }

            IConnectableLayer* weightsLayer = data.m_Network->AddConstantLayer(groupWeights);
            IConnectableLayer* biasLayer = data.m_Network->AddConstantLayer(groupBiases);
            IConnectableLayer* convLayer = data.m_Network->AddConvolution2dLayer(desc);

            convLayer->SetBackendId(setBackendConv);

            if (!convLayer)
            {
                return Fail("%s: AddConvolution2dLayer failed", __func__);
            }

            splitterLayer->GetOutputSlot(group).Connect(convLayer->GetInputSlot(0));
            weightsLayer->GetOutputSlot(0).Connect(convLayer->GetInputSlot(1));
            biasLayer->GetOutputSlot(0).Connect(convLayer->GetInputSlot(2));

            weightsLayer->GetOutputSlot(0).SetTensorInfo(groupWeightsInfo);
            biasLayer->GetOutputSlot(0).SetTensorInfo(groupBiasesInfo);
            convLayer->GetOutputSlot(0).SetTensorInfo(groupOutputInfo);

            if(isDynamic)
            {
                convLayer->GetOutputSlot(0).IsTensorInfoSet();

                validateFunc(convLayer->GetOutputSlot(0).GetTensorInfo(), isSupported);

                outputInfo = convLayer->GetOutputSlot(0).GetTensorInfo();

                if (!isSupported)
                {
                    return false;
                }
            }

            convLayers[index] = convLayer;
        }
    }

    //
    // Set up Concat layer
    //
    ConcatDescriptor concatDescriptor;
    // Equivalent to outputShape[channelsIndex], but we can't know the outputShape in the case of dynamic tensors
    concatDescriptor = ConcatDescriptor(weightsShape[0]);
    for (unsigned int group = 0u; group < numGroups; ++group)
    {
        for (unsigned int m = 0u; m < channelMultiplier; ++m)
        {
            auto index = group * channelMultiplier + m;
            concatDescriptor.SetViewOriginCoord(index, channelsIndex, index);
            concatDescriptor.SetConcatAxis(channelsIndex);
        }
    }

    isSupported = false;
    armnn::BackendId setBackendConcat;
    FORWARD_LAYER_SUPPORT_FUNC(__func__,
                               IsConcatSupported,
                               data.m_Backends,
                               isSupported,
                               setBackendConcat,
                               std::vector<const TensorInfo*>(numGroups * channelMultiplier, &groupOutputInfo),
                               outputInfo,
                               concatDescriptor);

    if (!isSupported)
    {
        return false;
    }

    IConnectableLayer* concatLayer = data.m_Network->AddConcatLayer(concatDescriptor);
    concatLayer->SetBackendId(setBackendConcat);
    if (!concatLayer)
    {
        return Fail("%s: AddConcatLayer failed", __func__);
    }

    for (unsigned int group = 0u; group < numGroups; ++group)
    {
        for (unsigned int m = 0u; m < channelMultiplier; ++m)
        {
            auto index = group * channelMultiplier + m;
            convLayers[index]->GetOutputSlot(0).Connect(concatLayer->GetInputSlot(index));
        }
    }
    concatLayer->GetOutputSlot(0).SetTensorInfo(outputInfo);

    return SetupAndTrackLayerOutputSlot(operation, 0, *concatLayer, model,
                                                   data, nullptr, nullptr, activation);
}

bool Converter::ConvertHardSwish(const Operation& operation, const Model& model, ConversionData& data)
{
    VLOG(DRIVER) << "Converter::ConvertHardSwish()";
    ActivationDescriptor desc;
    desc.m_Function = ActivationFunction::HardSwish;

    return ::ConvertToActivation(operation, __func__, desc, model, data);
}

bool Converter::ConvertInstanceNormalization(const Operation& operation, const Model& model, ConversionData& data)
{
    VLOG(DRIVER) << "Converter::ConvertInstanceNormalization()";

    LayerInputHandle input = ConvertToLayerInputHandle(operation, 0, model, data);
    if (!input.IsValid())
    {
        return Fail("%s: Operation has an invalid input 0", __func__);
    }

    const Operand* output = GetOutputOperand(operation, 0, model);
    if (!output)
    {
        return Fail("%s: Operation has an invalid output", __func__);
    }

    const TensorInfo& outputInfo = GetTensorInfoForOperand(*output);

    // Determine data type of input tensor
    OperandType inputType;
    if (!GetOperandType(operation, 0, model, inputType))
    {
        return Fail("%s: Operation has invalid inputs", __func__);
    }

    InstanceNormalizationDescriptor desc;

    // Read gamma, beta & epsilon
    if (inputType == OperandType::TENSOR_FLOAT16)
    {
        Half fp16Gamma;
        Half fp16Beta;
        Half fp16Epsilon;

        if (!GetInputScalar(operation, 1, OperandType::FLOAT16, fp16Gamma, model, data) ||
            !GetInputScalar(operation, 2, OperandType::FLOAT16, fp16Beta, model, data) ||
            !GetInputScalar(operation, 3, OperandType::FLOAT16, fp16Epsilon, model, data))
        {
            return Fail("%s: Operation has invalid inputs (FLOAT16)", __func__);
        }

        desc.m_Gamma = static_cast<float>(fp16Gamma);
        desc.m_Beta  = static_cast<float>(fp16Beta);
        desc.m_Eps   = static_cast<float>(fp16Epsilon);
    }
    else if (inputType == OperandType::TENSOR_FLOAT32)
    {
        if (!GetInputScalar(operation, 1, OperandType::FLOAT32, desc.m_Gamma, model, data) ||
            !GetInputScalar(operation, 2, OperandType::FLOAT32, desc.m_Beta, model, data) ||
            !GetInputScalar(operation, 3, OperandType::FLOAT32, desc.m_Eps, model, data))
        {
            return Fail("%s: Operation has invalid inputs (FLOAT32)", __func__);
        }
    }
    else
    {
        return Fail("%s: Unsupported input tensor type: %d", __func__, inputType);
    }

    desc.m_DataLayout = OptionalDataLayout(operation, 4, model, data);

    bool isSupported = false;
    armnn::BackendId setBackend;
    auto validateFunc = [&](const armnn::TensorInfo& outputInfo, bool& isSupported)
    {
        FORWARD_LAYER_SUPPORT_FUNC(__func__,
                                   IsInstanceNormalizationSupported,
                                   data.m_Backends,
                                   isSupported,
                                   setBackend,
                                   input.GetTensorInfo(),
                                   outputInfo,
                                   desc);
    };

    if(IsDynamicTensor(outputInfo))
    {
        isSupported = AreDynamicTensorsSupported();
    }
    else
    {
        validateFunc(outputInfo, isSupported);
    }

    if (!isSupported)
    {
        return false;
    }

    IConnectableLayer* layer = data.m_Network->AddInstanceNormalizationLayer(desc);
    layer->SetBackendId(setBackend);
    input.Connect(layer->GetInputSlot(0));

    return SetupAndTrackLayerOutputSlot(operation, 0, *layer, model, data, nullptr, validateFunc);
}

bool Converter::ConvertL2Normalization(const Operation& operation, const Model& model, ConversionData& data)
{
    VLOG(DRIVER) << "Converter::ConvertL2Normalization()";

    if (operation.inputs.size() != 1)
    {
        return Fail("%s: Optional inputs are not supported", __func__);
    }

    LayerInputHandle input = ConvertToLayerInputHandle(operation, 0, model, data);
    if (!input.IsValid())
    {
        return Fail("%s: Operation has invalid inputs", __func__);
    }

    const Operand* output = GetOutputOperand(operation, 0, model);
    if (!output)
    {
        return Fail("%s: Could not read output 0", __func__);
    }

    const armnn::TensorInfo& inputInfo  = input.GetTensorInfo();
    const armnn::TensorInfo& outputInfo = GetTensorInfoForOperand(*output);

    if (outputInfo.GetNumDimensions() != 4u)
    {
        return Fail("%s: Tensor Rank other than 4 is not supported", __func__);
    }

    armnn::L2NormalizationDescriptor desc;
    desc.m_DataLayout = armnn::DataLayout::NHWC;

    bool isSupported = false;
    armnn::BackendId setBackend;
    auto validateFunc = [&](const armnn::TensorInfo& outputInfo, bool& isSupported)
    {
        FORWARD_LAYER_SUPPORT_FUNC(__func__,
                                   IsL2NormalizationSupported,
                                   data.m_Backends,
                                   isSupported,
                                   setBackend,
                                   inputInfo,
                                   outputInfo,
                                   desc);
    };

    if(!IsDynamicTensor(outputInfo))
    {
        validateFunc(outputInfo, isSupported);
    }
    else
    {
        isSupported = AreDynamicTensorsSupported();
    }

    if (!isSupported)
    {
        return false;
    }

    armnn::IConnectableLayer* layer = data.m_Network->AddL2NormalizationLayer(desc);
    layer->SetBackendId(setBackend);
    assert(layer != nullptr);
    input.Connect(layer->GetInputSlot(0));

    return SetupAndTrackLayerOutputSlot(operation, 0, *layer, model, data, nullptr, validateFunc);
}

bool Converter::ConvertL2Pool2d(const Operation& operation, const Model& model, ConversionData& data)
{
    VLOG(DRIVER) << "Converter::ConvertL2Pool2d()";
    return ConvertPooling2d(operation, __func__, PoolingAlgorithm::L2, model, data);
}

bool Converter::ConvertLocalResponseNormalization(const Operation& operation,
                                                  const Model& model,
                                                  ConversionData& data)
{
    VLOG(DRIVER) << "Converter::ConvertLocalResponseNormalization()";

    if (operation.inputs.size() != 5)
    {
        return Fail("%s: Optional inputs are not supported", __func__);
    }

    LayerInputHandle input = ConvertToLayerInputHandle(operation, 0, model, data);
    if (!input.IsValid())
    {
        return Fail("%s: Operation has invalid inputs", __func__);
    }

    const Operand* output = GetOutputOperand(operation, 0, model);
    if (!output)
    {
        return Fail("%s: Could not read output 0", __func__);
    }

    const armnn::TensorInfo& inputInfo  = input.GetTensorInfo();
    const armnn::TensorInfo& outputInfo = GetTensorInfoForOperand(*output);

    if (outputInfo.GetNumDimensions() != 4u)
    {
        return Fail("%s: Tensor Rank other than 4 is not supported", __func__);
    }

    armnn::NormalizationDescriptor descriptor;
    descriptor.m_DataLayout      = armnn::DataLayout::NHWC;
    descriptor.m_NormChannelType = armnn::NormalizationAlgorithmChannel::Across;
    descriptor.m_NormMethodType  = armnn::NormalizationAlgorithmMethod::LocalBrightness;

    if (!input.IsValid() ||
        !GetInputScalar(operation, 1, OperandType::INT32, descriptor.m_NormSize, model, data) ||
        !GetInputFloat32(operation, 2, descriptor.m_K, model, data) ||
        !GetInputFloat32(operation, 3, descriptor.m_Alpha, model, data) ||
        !GetInputFloat32(operation, 4, descriptor.m_Beta, model, data))
    {
        return Fail("%s: Operation has invalid inputs", __func__);
    }

    // ArmNN expects normSize to be the full size of the normalization
    // window rather than the radius as in AndroidNN.
    descriptor.m_NormSize = 1 + (2 * descriptor.m_NormSize);

    bool isSupported = false;
    armnn::BackendId setBackend;
    auto validateFunc = [&](const armnn::TensorInfo& outputInfo, bool& isSupported)
    {
        FORWARD_LAYER_SUPPORT_FUNC(__func__,
                                   IsNormalizationSupported,
                                   data.m_Backends,
                                   isSupported,
                                   setBackend,
                                   inputInfo,
                                   outputInfo,
                                   descriptor);
    };

    if(!IsDynamicTensor(outputInfo))
    {
        validateFunc(outputInfo, isSupported);
    }
    else
    {
        isSupported = AreDynamicTensorsSupported();
    }

    if (!isSupported)
    {
        return false;
    }


    armnn::IConnectableLayer* layer = data.m_Network->AddNormalizationLayer(descriptor);
    layer->SetBackendId(setBackend);
    assert(layer != nullptr);
    input.Connect(layer->GetInputSlot(0));

    return SetupAndTrackLayerOutputSlot(operation, 0, *layer, model, data, nullptr, validateFunc);
}

bool Converter::ConvertLogicalBinary(const Operation& operation,
                                     const Model& model,
                                     ConversionData& data,
                                     armnn::LogicalBinaryOperation logicalOperation)
{
    VLOG(DRIVER) << "Converter::ConvertLogicalBinary()";
    VLOG(DRIVER) << "ConvertLogicalBinary()";
    VLOG(DRIVER) << "logicalOperation = " << GetLogicalBinaryOperationAsCString(logicalOperation);

    LayerInputHandle input0 = ConvertToLayerInputHandle(operation, 0, model, data);
    LayerInputHandle input1 = ConvertToLayerInputHandle(operation, 1, model, data);

    if (!(input0.IsValid() && input1.IsValid()))
    {
        return Fail("%s: Operation has invalid inputs", __func__);
    }

    const Operand* output = GetOutputOperand(operation, 0, model);
    if (!output)
    {
        return Fail("%s: Could not read output 0", __func__);
    }

    const TensorInfo& inputInfo0 = input0.GetTensorInfo();
    const TensorInfo& inputInfo1 = input1.GetTensorInfo();
    const TensorInfo& outputInfo = GetTensorInfoForOperand(*output);

    LogicalBinaryDescriptor descriptor(logicalOperation);

    bool isSupported = false;
    armnn::BackendId setBackend;
    auto validateFunc = [&](const armnn::TensorInfo& outputInfo, bool& isSupported)
    {
        FORWARD_LAYER_SUPPORT_FUNC(__func__,
                                   IsLogicalBinarySupported,
                                   data.m_Backends,
                                   isSupported,
                                   setBackend,
                                   inputInfo0,
                                   inputInfo1,
                                   outputInfo,
                                   descriptor);
    };

    if(!IsDynamicTensor(outputInfo))
    {
        validateFunc(outputInfo, isSupported);
    }
    else
    {
        isSupported = AreDynamicTensorsSupported();
    }

    if (!isSupported)
    {
        return false;
    }

    IConnectableLayer* layer = data.m_Network->AddLogicalBinaryLayer(descriptor);
    layer->SetBackendId(setBackend);
    assert(layer != nullptr);

    bool isReshapeSupported = BroadcastTensor(input0, input1, layer, data);
    if (!isReshapeSupported)
    {
        return false;
    }

    return SetupAndTrackLayerOutputSlot(operation, 0, *layer, model, data, nullptr, validateFunc);
}

bool Converter::ConvertLogistic(const Operation& operation, const Model& model, ConversionData& data)
{
    VLOG(DRIVER) << "Converter::ConvertLogistic()";
    armnn::ActivationDescriptor desc;
    desc.m_Function = armnn::ActivationFunction::Sigmoid;

    return ConvertToActivation(operation, __func__, desc, model, data);
}

bool Converter::ConvertLogSoftmax(const Operation& operation, const Model& model, ConversionData& data)
{
    VLOG(DRIVER) << "Converter::ConvertLogSoftmax()";

    LayerInputHandle input = ConvertToLayerInputHandle(operation, 0, model, data);
    if (!input.IsValid())
    {
        return Fail("%s: Failed to read input 0", __func__);
    }

    const Operand* output = GetOutputOperand(operation, 0, model);
    if (!output)
    {
        return Fail("%s: Failed to read output", __func__);
    }

    const TensorInfo& outputInfo = GetTensorInfoForOperand(*output);

    // Determine data type of input tensor
    OperandType inputType;
    if (!GetOperandType(operation, 0, model, inputType))
    {
        return Fail("%s: Operation has invalid inputs", __func__);
    }

    LogSoftmaxDescriptor descriptor;

    // Read beta
    if (inputType == OperandType::TENSOR_FLOAT16)
    {
        Half fp16Beta;
        if (!GetInputScalar(operation, 1, OperandType::FLOAT16, fp16Beta, model, data))
        {
            return Fail("%s: Failed to read input 1 (FLOAT16)", __func__);
        }

        descriptor.m_Beta  = static_cast<float>(fp16Beta);
    }
    else if (inputType == OperandType::TENSOR_FLOAT32)
    {
        if (!GetInputScalar(operation, 1, OperandType::FLOAT32, descriptor.m_Beta, model, data))
        {
            return Fail("%s: Failed to read input 1 (FLOAT32)", __func__);
        }
    }
    else
    {
        return Fail("%s: Unsupported input tensor type: %d", __func__, inputType);
    }

    // Read axis
    if (!GetInputInt32(operation, 2, descriptor.m_Axis, model, data))
    {
        return Fail("%s: Failed to read input 2", __func__);
    }

    bool isSupported = false;
    armnn::BackendId setBackend;
    auto validateFunc = [&](const armnn::TensorInfo& outputInfo, bool& isSupported)
    {
        FORWARD_LAYER_SUPPORT_FUNC(__func__,
                                   IsLogSoftmaxSupported,
                                   data.m_Backends,
                                   isSupported,
                                   setBackend,
                                   input.GetTensorInfo(),
                                   outputInfo,
                                   descriptor);
    };

    if(IsDynamicTensor(outputInfo))
    {
        isSupported = AreDynamicTensorsSupported();
    }
    else
    {
        validateFunc(outputInfo, isSupported);
    }

    if (!isSupported)
    {
        return false;
    }

    IConnectableLayer* layer = data.m_Network->AddLogSoftmaxLayer(descriptor);
    layer->SetBackendId(setBackend);
    if (!layer)
    {
        return Fail("%s: AddLogSoftmaxLayer() returned nullptr", __func__);
    }

    input.Connect(layer->GetInputSlot(0));

    return SetupAndTrackLayerOutputSlot(operation, 0, *layer, model, data, nullptr, validateFunc);
}

bool Converter::ConvertLstm(const Operation& operation, const Model& model, ConversionData& data)
{
    VLOG(DRIVER) << "Converter::ConvertLstm()";

    // Inputs:
    // 00: The input: A 2-D tensor of ANEURALNETWORKS_TENSOR_FLOAT32, of shape [batch_size, input_size], where
    //      “batch_size” corresponds to the batching dimension, and “input_size” is the size of the input.
    LayerInputHandle input = ConvertToLayerInputHandle(operation, 0, model, data);
    if (!input.IsValid())
    {
        return Fail("%s: Could not read input 0: input", __func__);
    }
    // 18: The output state: A 2-D tensor of ANEURALNETWORKS_TENSOR_FLOAT32, of shape [batch_size, output_size].
    LayerInputHandle outputStateIn = ConvertToLayerInputHandle(operation, 18, model, data);
    if (!outputStateIn.IsValid())
    {
        return Fail("%s: Could not read input 18: outputStateIn", __func__);
    }
    // 19: The cell state: A 2-D tensor of ANEURALNETWORKS_TENSOR_FLOAT32, of shape [batch_size, num_units].
    LayerInputHandle cellStateIn = ConvertToLayerInputHandle(operation, 19, model, data);
    if (!cellStateIn.IsValid())
    {
        return Fail("%s: Could not read input 19: cellStateIn", __func__);
    }

    // Get the mandatory input tensors:
    // 02: The input-to-forget weights: A 2-D tensor of ANEURALNETWORKS_TENSOR_FLOAT32, of shape
    //     [num_units, input_size].
    const ConstTensorPin inputToForgetWeightsPin =
        (DequantizeAndMakeConstTensorPin(operation, model, data, 2));
    // 03: The input-to-cell weights: A 2-D tensor of ANEURALNETWORKS_TENSOR_FLOAT32, of shape
    // [num_units, input_size].
    const ConstTensorPin inputToCellWeightsPin =
        (DequantizeAndMakeConstTensorPin(operation, model, data, 3));
    // 04: The input-to-output weights: A 2-D tensor of ANEURALNETWORKS_TENSOR_FLOAT32, of shape
    //     [num_units, input_size].
    const ConstTensorPin inputToOutputWeightsPin =
        (DequantizeAndMakeConstTensorPin(operation, model, data, 4));
    // 06: The recurrent-to-forget weights: A 2-D tensor of ANEURALNETWORKS_TENSOR_FLOAT32, of shape
    //     [num_units, output_size].
    const ConstTensorPin recurrentToForgetWeightsPin =
        (DequantizeAndMakeConstTensorPin(operation, model, data, 6));
    // 07: The recurrent-to-cell weights: A 2-D tensor of ANEURALNETWORKS_TENSOR_FLOAT32, of shape
    //     [num_units, output_size].
    const ConstTensorPin recurrentToCellWeightsPin =
        (DequantizeAndMakeConstTensorPin(operation, model, data, 7));
    // 08: The recurrent-to-output weights: A 2-D tensor of ANEURALNETWORKS_TENSOR_FLOAT32, of shape
    //     [num_units, output_size].
    const ConstTensorPin recurrentToOutputWeightsPin =
        (DequantizeAndMakeConstTensorPin(operation, model, data, 8));
    // 13: The forget gate bias: A 1-D tensor of ANEURALNETWORKS_TENSOR_FLOAT32, of shape [num_units].
    const ConstTensorPin forgetGateBiasPin =
        ConvertOperationInputToConstTensorPin(operation, 13, model, data);
    // 14: The cell bias: A 1-D tensor of ANEURALNETWORKS_TENSOR_FLOAT32, of shape [num_units].
    const ConstTensorPin cellBiasPin =
        ConvertOperationInputToConstTensorPin(operation, 14, model, data);
    // 15: The output gate bias: A 1-D tensor of ANEURALNETWORKS_TENSOR_FLOAT32, of shape [num_units].
    const ConstTensorPin outputGateBiasPin =
        ConvertOperationInputToConstTensorPin(operation, 15, model, data);

    if (!inputToForgetWeightsPin.IsValid() ||
        !inputToCellWeightsPin.IsValid() ||
        !inputToOutputWeightsPin.IsValid() ||
        !recurrentToForgetWeightsPin.IsValid() ||
        !recurrentToCellWeightsPin.IsValid() ||
        !recurrentToOutputWeightsPin.IsValid() ||
        !forgetGateBiasPin.IsValid() ||
        !cellBiasPin.IsValid() ||
        !outputGateBiasPin.IsValid())
    {
        return Fail("%s: Operation has invalid tensor inputs", __func__);
    }

    // Get the optional input tensors:
    // 01: The input-to-input weights: Optional. A 2-D tensor of ANEURALNETWORKS_TENSOR_FLOAT32, of shape
    //     [num_units, input_size], where “num_units” corresponds to the number of cell units.
    const ConstTensorPin inputToInputWeightsPin =
        (DequantizeAndMakeConstTensorPin(operation, model, data, 1, true));
    // 05: The recurrent-to-input weights: Optional. A 2-D tensor of ANEURALNETWORKS_TENSOR_FLOAT32, of shape
    //     [num_units, output_size], where “output_size” corresponds to either the number of cell units (i.e.,
    //     “num_units”), or the second dimension of the “projection_weights”, if defined.
    const ConstTensorPin recurrentToInputWeightsPin =
        (DequantizeAndMakeConstTensorPin(operation, model, data, 5, true));
    // 09: The cell-to-input weights: Optional. A 1-D tensor of ANEURALNETWORKS_TENSOR_FLOAT32, of shape [num_units].
    const ConstTensorPin cellToInputWeightsPin =
        (DequantizeAndMakeConstTensorPin(operation, model, data, 9, true));
    // 10: The cell-to-forget weights: Optional. A 1-D tensor of ANEURALNETWORKS_TENSOR_FLOAT32, of shape [num_units].
    const ConstTensorPin cellToForgetWeightsPin =
        (DequantizeAndMakeConstTensorPin(operation, model, data, 10, true));
    // 11: The cell-to-output weights: Optional. A 1-D tensor of ANEURALNETWORKS_TENSOR_FLOAT32, of shape [num_units].
    const ConstTensorPin cellToOutputWeightsPin =
        (DequantizeAndMakeConstTensorPin(operation, model, data, 11, true));
    // 12: The input gate bias: Optional. A 1-D tensor of ANEURALNETWORKS_TENSOR_FLOAT32, of shape [num_units].
    const ConstTensorPin inputGateBiasPin =
        ConvertOperationInputToConstTensorPin(operation,
                                                         12,
                                                         model,
                                                         data,
                                                         g_DontPermute,
                                                         nullptr,
                                                         true);

    // 16: The projection weights: Optional. A 2-D tensor of ANEURALNETWORKS_TENSOR_FLOAT32, of shape
    //     [output_size, num_units].
    const ConstTensorPin projectionWeightsPin =
        (DequantizeAndMakeConstTensorPin(operation, model, data, 16, true));
    // 17: The projection bias: Optional. A 1-D tensor of ANEURALNETWORKS_TENSOR_FLOAT32, of shape [output_size].
    const ConstTensorPin projectionBiasPin =
        ConvertOperationInputToConstTensorPin(operation,
                                                         17,
                                                         model,
                                                         data,
                                                         g_DontPermute,
                                                         nullptr,
                                                         true);

    if ((!inputToInputWeightsPin.IsValid() && !inputToInputWeightsPin.IsOptional()) ||
        (!recurrentToInputWeightsPin.IsValid() && !recurrentToInputWeightsPin.IsOptional()) ||
        (!cellToInputWeightsPin.IsValid() && !cellToInputWeightsPin.IsOptional()) ||
        (!cellToForgetWeightsPin.IsValid() && !cellToForgetWeightsPin.IsOptional()) ||
        (!cellToOutputWeightsPin.IsValid() && !cellToOutputWeightsPin.IsOptional()) ||
        (!inputGateBiasPin.IsValid() && !inputGateBiasPin.IsOptional()) ||
        (!projectionWeightsPin.IsValid() && !projectionWeightsPin.IsOptional()) ||
        (!projectionBiasPin.IsValid() && !projectionBiasPin.IsOptional()))
    {
        return Fail("%s: Operation has invalid tensor inputs", __func__);
    }

    // Get the mandatory input scalars (actually 1-D tensors of size 1):
    // 20: The activation function: A value indicating the activation function:
    //     0: None; 1: Relu; 3: Relu6; 4: Tanh; 6: Sigmoid.
    // 21: The clipping threshold: for the cell state, such that values are bound within [-cell_clip, cell_clip].
    //     If set to 0.0 then clipping is disabled.
    // 22: The clipping threshold: for the output from the projection layer, such that values are bound within
    //     [-proj_clip, proj_clip]. If set to 0.0 then clipping is disabled.
    ActivationFn activation = ActivationFn::kActivationNone;
    float cellClip;
    float projClip;
    if (!GetInputActivationFunctionFromTensor(operation, 20, activation, model, data) ||
        !GetInputScalar(operation, 21, OperandType::FLOAT32, cellClip, model, data) ||
        !GetInputScalar(operation, 22, OperandType::FLOAT32, projClip, model, data))
    {
        return Fail("%s: Operation has invalid scalar inputs", __func__);
    }

    // Get the normalization tensors
    // 23: The input layer normalization weights. A 1-D tensor of shape [num_units].
    //     Used to rescale normalized inputs to activation at input gate.
    const ConstTensorPin inputLayerNormWeightsPin
        (DequantizeAndMakeConstTensorPin(operation, model, data, 23, true));

    // 24: The forget layer normalization weights. A 1-D tensor of shape [num_units].
    //     Used to rescale normalized inputs to activation at forget gate.
    const ConstTensorPin forgetLayerNormWeightsPin =
        ConvertOperationInputToConstTensorPin(operation,
                                                        24,
                                                        model,
                                                        data,
                                                        g_DontPermute,
                                                        nullptr,
                                                        true);

    // 25: The cell layer normalization weights. A 1-D tensor of shape [num_units].
    //     Used to rescale normalized inputs to activation at cell gate.
    const ConstTensorPin cellLayerNormWeightsPin =
        ConvertOperationInputToConstTensorPin(operation,
                                                         25,
                                                         model,
                                                         data,
                                                         g_DontPermute,
                                                         nullptr,
                                                         true);

    // 26: The output layer normalization weights. A 1-D tensor of shape [num_units].
    //     Used to rescale normalized inputs to activation at output gate.
    const ConstTensorPin outputLayerNormWeightsPin =
        ConvertOperationInputToConstTensorPin(operation,
                                                         26,
                                                         model,
                                                         data,
                                                         g_DontPermute,
                                                         nullptr,
                                                         true);

    // Outputs:
    // 00: The scratch buffer: A 2-D tensor of ANEURALNETWORKS_TENSOR_FLOAT32, of shape [batch_size, num_units * 4]
    // with CIFG, or [batch_size, num_units * 3] without CIFG.
    const Operand* scratchBuffer = GetOutputOperand(operation, 0, model);
    if (!scratchBuffer)
    {
        return Fail("%s: Could not read output 0: scratchBuffer", __func__);
    }
    // 01: The output state (out): A 2-D tensor of ANEURALNETWORKS_TENSOR_FLOAT32, of shape [batch_size, output_size].
    const Operand* outputStateOut = GetOutputOperand(operation, 1, model);
    if (!outputStateOut)
    {
        return Fail("%s: Could not read output 1: outputStateOut", __func__);
    }
    // 02: The cell state (out): A 2-D tensor of ANEURALNETWORKS_TENSOR_FLOAT32, of shape [batch_size, num_units].
    const Operand* cellStateOut = GetOutputOperand(operation, 2, model);
    if (!cellStateOut)
    {
        return Fail("%s: Could not read output 2: cellStateOut", __func__);
    }
    // 03: The output: A 2-D tensor of ANEURALNETWORKS_TENSOR_FLOAT32, of shape [batch_size, output_size]. This is
    //     effectively the same as the current “output state (out)” value.
    const Operand* output = GetOutputOperand(operation, 3, model);
    if (!output)
    {
        return Fail("%s: Could not read output 3: output", __func__);
    }

    // set the params structure for the AddLstmLayer call
    LstmInputParams params;
    params.m_InputToInputWeights = inputToInputWeightsPin.GetConstTensorPtr();
    params.m_InputToForgetWeights = inputToForgetWeightsPin.GetConstTensorPtr();
    params.m_InputToCellWeights = inputToCellWeightsPin.GetConstTensorPtr();
    params.m_InputToOutputWeights = inputToOutputWeightsPin.GetConstTensorPtr();
    params.m_RecurrentToInputWeights = recurrentToInputWeightsPin.GetConstTensorPtr();
    params.m_RecurrentToForgetWeights = recurrentToForgetWeightsPin.GetConstTensorPtr();
    params.m_RecurrentToCellWeights = recurrentToCellWeightsPin.GetConstTensorPtr();
    params.m_RecurrentToOutputWeights = recurrentToOutputWeightsPin.GetConstTensorPtr();
    params.m_CellToInputWeights = cellToInputWeightsPin.GetConstTensorPtr();
    params.m_CellToForgetWeights = cellToForgetWeightsPin.GetConstTensorPtr();
    params.m_CellToOutputWeights = cellToOutputWeightsPin.GetConstTensorPtr();
    params.m_InputGateBias = inputGateBiasPin.GetConstTensorPtr();
    params.m_ForgetGateBias = forgetGateBiasPin.GetConstTensorPtr();
    params.m_CellBias = cellBiasPin.GetConstTensorPtr();
    params.m_OutputGateBias = outputGateBiasPin.GetConstTensorPtr();
    params.m_ProjectionWeights = projectionWeightsPin.GetConstTensorPtr();
    params.m_ProjectionBias = projectionBiasPin.GetConstTensorPtr();
    params.m_InputLayerNormWeights = inputLayerNormWeightsPin.GetConstTensorPtr();
    params.m_ForgetLayerNormWeights = forgetLayerNormWeightsPin.GetConstTensorPtr();
    params.m_CellLayerNormWeights = cellLayerNormWeightsPin.GetConstTensorPtr();
    params.m_OutputLayerNormWeights = outputLayerNormWeightsPin.GetConstTensorPtr();

    // set the layer descriptor
    LstmDescriptor desc;
    desc.m_ActivationFunc = activation;
    desc.m_ClippingThresCell = cellClip;
    desc.m_ClippingThresProj = projClip;
    desc.m_CifgEnabled = (params.m_InputToInputWeights == nullptr ||
                          params.m_RecurrentToInputWeights == nullptr ||
                          params.m_InputGateBias == nullptr);
    desc.m_PeepholeEnabled = (params.m_CellToForgetWeights != nullptr ||
                              params.m_CellToOutputWeights != nullptr);
    desc.m_ProjectionEnabled = (params.m_ProjectionWeights != nullptr);
    desc.m_LayerNormEnabled = (params.m_InputLayerNormWeights != nullptr ||
                               params.m_ForgetLayerNormWeights != nullptr ||
                               params.m_CellLayerNormWeights != nullptr ||
                               params.m_OutputLayerNormWeights != nullptr);

    // validate the optional input groups
    if (desc.m_CifgEnabled &&
        (params.m_InputToInputWeights != nullptr ||
         params.m_RecurrentToInputWeights != nullptr ||
         params.m_InputGateBias != nullptr))
    {
        return Fail("%s: All, or none, of input-to-input weights, recurrent-to-input weights,"
                    " and input gate bias must be provided", __func__);
    }

    if (!desc.m_ProjectionEnabled && params.m_ProjectionBias != nullptr)
    {
        return Fail("%s: projection bias should not be provided without projection weights", __func__);
    }

    if (desc.m_PeepholeEnabled &&
        (params.m_CellToForgetWeights == nullptr ||
         params.m_CellToOutputWeights == nullptr ||
         (!desc.m_CifgEnabled && params.m_CellToInputWeights == nullptr)))
    {
        return Fail("%s: All, or none, of cell-to-forget weights and cell-to-output weights must be provided"
                    " and, if CIFG is not enabled, cell-to-input weights must also be provided", __func__);
    }

    if (desc.m_LayerNormEnabled &&
        (params.m_ForgetLayerNormWeights == nullptr ||
         params.m_CellLayerNormWeights == nullptr ||
         params.m_OutputLayerNormWeights == nullptr ||
         (!desc.m_CifgEnabled && params.m_InputLayerNormWeights == nullptr)))
    {
        return Fail("%s: All, or none, of forget-norm weights, cell-norm weights and output-norm weights must be"
                    " provided and, if CIFG is not enabled, input-norm weights must also be provided", __func__);
    }

    // Check if the layer is supported
    // Inputs
    const TensorInfo& inputInfo         = input.GetTensorInfo();
    const TensorInfo& outputStateInInfo = outputStateIn.GetTensorInfo();
    const TensorInfo& cellStateInInfo   = cellStateIn.GetTensorInfo();

    // Outputs
    const TensorInfo& scratchBufferInfo  = GetTensorInfoForOperand(*scratchBuffer);
    const TensorInfo& outputStateOutInfo = GetTensorInfoForOperand(*outputStateOut);
    const TensorInfo& cellStateOutInfo   = GetTensorInfoForOperand(*cellStateOut);
    const TensorInfo& outputInfo         = GetTensorInfoForOperand(*output);

    // Basic parameters
    LstmInputParamsInfo paramsInfo;
    paramsInfo.m_InputToForgetWeights     = &(params.m_InputToForgetWeights->GetInfo());
    paramsInfo.m_InputToCellWeights       = &(params.m_InputToCellWeights->GetInfo());
    paramsInfo.m_InputToOutputWeights     = &(params.m_InputToOutputWeights->GetInfo());
    paramsInfo.m_RecurrentToForgetWeights = &(params.m_RecurrentToForgetWeights->GetInfo());
    paramsInfo.m_RecurrentToCellWeights   = &(params.m_RecurrentToCellWeights->GetInfo());
    paramsInfo.m_RecurrentToOutputWeights = &(params.m_RecurrentToOutputWeights->GetInfo());
    paramsInfo.m_ForgetGateBias           = &(params.m_ForgetGateBias->GetInfo());
    paramsInfo.m_CellBias                 = &(params.m_CellBias->GetInfo());
    paramsInfo.m_OutputGateBias           = &(params.m_OutputGateBias->GetInfo());

    // Optional parameters
    if (!desc.m_CifgEnabled)
    {
        paramsInfo.m_InputToInputWeights = &(params.m_InputToInputWeights->GetInfo());
        paramsInfo.m_RecurrentToInputWeights = &(params.m_RecurrentToInputWeights->GetInfo());
        if (params.m_CellToInputWeights != nullptr)
        {
            paramsInfo.m_CellToInputWeights = &(params.m_CellToInputWeights->GetInfo());
        }
        paramsInfo.m_InputGateBias = &(params.m_InputGateBias->GetInfo());
    }

    if (desc.m_ProjectionEnabled)
    {
        paramsInfo.m_ProjectionWeights = &(params.m_ProjectionWeights->GetInfo());
        if (params.m_ProjectionBias != nullptr)
        {
            paramsInfo.m_ProjectionBias = &(params.m_ProjectionBias->GetInfo());
        }
    }

    if (desc.m_PeepholeEnabled)
    {
        paramsInfo.m_CellToForgetWeights = &(params.m_CellToForgetWeights->GetInfo());
        paramsInfo.m_CellToOutputWeights = &(params.m_CellToOutputWeights->GetInfo());
    }

    if (desc.m_LayerNormEnabled)
    {
        if(!desc.m_CifgEnabled)
        {
            paramsInfo.m_InputLayerNormWeights = &(params.m_InputLayerNormWeights->GetInfo());
        }
        paramsInfo.m_ForgetLayerNormWeights = &(params.m_ForgetLayerNormWeights->GetInfo());
        paramsInfo.m_CellLayerNormWeights = &(params.m_CellLayerNormWeights->GetInfo());
        paramsInfo.m_OutputLayerNormWeights = &(params.m_OutputLayerNormWeights->GetInfo());
    }

    bool isSupported = false;
    armnn::BackendId setBackend;
    auto validateFunc = [&](const armnn::TensorInfo& outputInfo, bool& isSupported)
    {
        FORWARD_LAYER_SUPPORT_FUNC(__func__,
                                   IsLstmSupported,
                                   data.m_Backends,
                                   isSupported,
                                   setBackend,
                                   inputInfo,
                                   outputStateInInfo,
                                   cellStateInInfo,
                                   scratchBufferInfo,
                                   outputStateOutInfo,
                                   cellStateOutInfo,
                                   outputInfo,
                                   desc,
                                   paramsInfo);
    };

    bool isDynamic = false;
    if (!IsDynamicTensor(outputStateOutInfo) &&
        !IsDynamicTensor(scratchBufferInfo)  &&
        !IsDynamicTensor(cellStateOutInfo)   &&
        !IsDynamicTensor(outputInfo))
    {
        validateFunc(outputInfo, isSupported);
    }
    else
    {
        isDynamic = true;
        isSupported = AreDynamicTensorsSupported();
    }

    if (!isSupported)
    {
        return false;
    }

    // Add the layer
    IConnectableLayer* layer = data.m_Network->AddLstmLayer(desc, params, "Lstm");
    layer->SetBackendId(setBackend);

    input.Connect(layer->GetInputSlot(0));
    outputStateIn.Connect(layer->GetInputSlot(1));
    cellStateIn.Connect(layer->GetInputSlot(2));

    if (!isDynamic)
    {
        return (
             SetupAndTrackLayerOutputSlot(operation, 0, *layer, 0, model, data) &&
             SetupAndTrackLayerOutputSlot(operation, 1, *layer, 1, model, data) &&
             SetupAndTrackLayerOutputSlot(operation, 2, *layer, 2, model, data) &&
             SetupAndTrackLayerOutputSlot(operation, 3, *layer, 3, model, data));
    }
    else
    {
        return (
             SetupAndTrackLayerOutputSlot(operation, 0, *layer, 0, model, data) &&
             SetupAndTrackLayerOutputSlot(operation, 1, *layer, 1, model, data) &&
             SetupAndTrackLayerOutputSlot(operation, 2, *layer, 2, model, data) &&
             SetupAndTrackLayerOutputSlot(
                 operation, 3, *layer, 3, model, data, nullptr, validateFunc, ActivationFn::kActivationNone, true));
    }

}

bool Converter::ConvertMaxPool2d(const Operation& operation, const Model& model, ConversionData& data)
{
    VLOG(DRIVER) << "Converter::ConvertMaxPool2d()";
    return ConvertPooling2d(operation, __func__, PoolingAlgorithm::Max, model, data);
}

bool Converter::ConvertMaximum(const Operation& operation, const Model& model, ConversionData& data)
{
    VLOG(DRIVER) << "Converter::ConvertMaximum()";

    LayerInputHandle input0 = ConvertToLayerInputHandle(operation, 0, model, data);
    LayerInputHandle input1 = ConvertToLayerInputHandle(operation, 1, model, data);

    if (!input0.IsValid() || !input1.IsValid())
    {
        return Fail("%s: Operation has invalid inputs", __func__);
    }

    const Operand* outputOperand = GetOutputOperand(operation, 0, model);
    if (!outputOperand)
    {
        return Fail("%s: Could not read output", __func__);
    }

    const TensorInfo& outInfo = GetTensorInfoForOperand(*outputOperand);

    bool isSupported = false;
    armnn::BackendId setBackend;
    auto validateFunc = [&](const armnn::TensorInfo& outInfo, bool& isSupported)
    {
        ARMNN_NO_DEPRECATE_WARN_BEGIN
        FORWARD_LAYER_SUPPORT_FUNC(__func__,
                                   IsMaximumSupported,
                                   data.m_Backends,
                                   isSupported,
                                   setBackend,
                                   input0.GetTensorInfo(),
                                   input1.GetTensorInfo(),
                                   outInfo);
        ARMNN_NO_DEPRECATE_WARN_END
    };

    if(IsDynamicTensor(outInfo))
    {
        isSupported = AreDynamicTensorsSupported();
    }
    else
    {
        validateFunc(outInfo, isSupported);
    }

    if (!isSupported)
    {
        return false;
    }

    ARMNN_NO_DEPRECATE_WARN_BEGIN
    IConnectableLayer* layer = data.m_Network->AddMaximumLayer();
    ARMNN_NO_DEPRECATE_WARN_END
    layer->SetBackendId(setBackend);
    assert(layer != nullptr);
    bool isReshapeSupported = BroadcastTensor(input0, input1, layer, data);
    if (!isReshapeSupported)
    {
        return false;
    }

    return SetupAndTrackLayerOutputSlot(operation, 0, *layer, model, data, nullptr, validateFunc);
}

bool Converter::ConvertMean(const Operation& operation, const Model& model, ConversionData& data)
{
    VLOG(DRIVER) << "Converter::ConvertMean()";

    LayerInputHandle input = ConvertToLayerInputHandle(operation, 0, model, data);
    if (!input.IsValid())
    {
        return Fail("%s: Operation has invalid inputs", __func__);
    }

    const Operand* output = GetOutputOperand(operation, 0, model);
    if (!output)
    {
        return Fail("%s: Could not read output 0", __func__);
    }

    const armnn::TensorInfo& outputInfo = GetTensorInfoForOperand(*output);

    const Operand* axisOperand = GetInputOperand(operation, 1, model);
    if (!axisOperand)
    {
        return Fail("%s: Could not read input 1", __func__);
    }

    std::vector<int32_t> axis;
    if (!GetTensorInt32Values(*axisOperand, axis, model, data))
    {
        return Fail("%s: Input 1 has invalid values", __func__);
    }

    const armnn::TensorInfo& inputInfo = input.GetTensorInfo();

    // Convert the axis to unsigned int and remove duplicates.
    unsigned int rank = inputInfo.GetNumDimensions();
    std::set<unsigned int> uniqueAxis;
    std::transform(axis.begin(), axis.end(),
                   std::inserter(uniqueAxis, uniqueAxis.begin()),
                   [rank](int i) -> unsigned int { return (i + rank) % rank; });

    // Get the "keep dims" flag.
    int32_t keepDims = 0;
    if (!GetInputInt32(operation, 2, keepDims, model, data))
    {
        return Fail("%s: Could not read input 2", __func__);
    }

    armnn::MeanDescriptor descriptor;
    descriptor.m_Axis.assign(uniqueAxis.begin(), uniqueAxis.end());
    descriptor.m_KeepDims = keepDims > 0;

    bool isSupported = false;
    armnn::BackendId setBackend;
    auto validateFunc = [&](const armnn::TensorInfo& outputInfo, bool& isSupported)
    {
        FORWARD_LAYER_SUPPORT_FUNC(__func__,
                                   IsMeanSupported,
                                   data.m_Backends,
                                   isSupported,
                                   setBackend,
                                   inputInfo,
                                   outputInfo,
                                   descriptor);
    };

    if(!IsDynamicTensor(outputInfo))
    {
        validateFunc(outputInfo, isSupported);
    }
    else
    {
        isSupported = AreDynamicTensorsSupported();
    }

    if (!isSupported)
    {
        return false;
    }

    armnn::IConnectableLayer* const layer = data.m_Network->AddMeanLayer(descriptor);
    layer->SetBackendId(setBackend);
    assert(layer != nullptr);
    input.Connect(layer->GetInputSlot(0));

    return SetupAndTrackLayerOutputSlot(operation, 0, *layer, model, data, nullptr, validateFunc);
}

bool Converter::ConvertMinimum(const Operation& operation, const Model& model, ConversionData& data)
{
    VLOG(DRIVER) << "Converter::ConvertMinimum()";

    LayerInputHandle input0 = ConvertToLayerInputHandle(operation, 0, model, data);
    LayerInputHandle input1 = ConvertToLayerInputHandle(operation, 1, model, data);

    if (!input0.IsValid() || !input1.IsValid())
    {
        return Fail("%s: Operation has invalid inputs", __func__);
    }

    const Operand* output = GetOutputOperand(operation, 0, model);
    if (!output)
    {
        return Fail("%s: Could not read output 0", __func__);
    }

    const TensorInfo& outputInfo = GetTensorInfoForOperand(*output);

    bool isSupported = false;
    armnn::BackendId setBackend;
    auto validateFunc = [&](const armnn::TensorInfo& outputInfo, bool& isSupported)
    {
        ARMNN_NO_DEPRECATE_WARN_BEGIN
        FORWARD_LAYER_SUPPORT_FUNC(__func__,
                                   IsMinimumSupported,
                                   data.m_Backends,
                                   isSupported,
                                   setBackend,
                                   input0.GetTensorInfo(),
                                   input1.GetTensorInfo(),
                                   outputInfo);
        ARMNN_NO_DEPRECATE_WARN_END
    };

    if(IsDynamicTensor(outputInfo))
    {
        isSupported = AreDynamicTensorsSupported();
    }
    else
    {
        validateFunc(outputInfo, isSupported);
    }

    if (!isSupported)
    {
        return false;
    }

    ARMNN_NO_DEPRECATE_WARN_BEGIN
    IConnectableLayer* const layer = data.m_Network->AddMinimumLayer();
    ARMNN_NO_DEPRECATE_WARN_END
    layer->SetBackendId(setBackend);
    assert(layer != nullptr);
    bool isReshapeSupported = BroadcastTensor(input0, input1, layer, data);
    if (!isReshapeSupported)
    {
        return false;
    }

    return SetupAndTrackLayerOutputSlot(operation, 0, *layer, model, data, nullptr, validateFunc);
}

bool Converter::ConvertMul(const Operation& operation, const Model& model, ConversionData& data)
{
    VLOG(DRIVER) << "Converter::ConvertMul()";

    LayerInputHandle input0 = ConvertToLayerInputHandle(operation, 0, model, data);
    LayerInputHandle input1 = ConvertToLayerInputHandle(operation, 1, model, data);

    if (!input0.IsValid() || !input1.IsValid())
    {
        return Fail("%s: Operation has invalid inputs", __func__);
    }

    // The FuseActivation parameter is always the input index 2
    // and it should be optional
    ActivationFn activationFunction;
    if (!GetOptionalInputActivation(operation, 2, activationFunction, model, data))
    {
        return Fail("%s: Operation has invalid inputs", __func__);
    }

    const Operand* outputOperand = GetOutputOperand(operation, 0, model);

    if (outputOperand == nullptr)
    {
        return false;
    }

    const armnn::TensorInfo& outputInfo = GetTensorInfoForOperand(*outputOperand);

    bool isSupported = false;
    armnn::BackendId setBackend;
    auto validateFunc = [&](const armnn::TensorInfo& outputInfo, bool& isSupported)
    {
        ARMNN_NO_DEPRECATE_WARN_BEGIN
        FORWARD_LAYER_SUPPORT_FUNC(__func__,
                                   IsMultiplicationSupported,
                                   data.m_Backends,
                                   isSupported,
                                   setBackend,
                                   input0.GetTensorInfo(),
                                   input1.GetTensorInfo(),
                                   outputInfo);
        ARMNN_NO_DEPRECATE_WARN_END
    };

    if(!IsDynamicTensor(outputInfo))
    {
        validateFunc(outputInfo, isSupported);
    }
    else
    {
        isSupported = AreDynamicTensorsSupported();
    }

    if (!isSupported)
    {
        return false;
    }

    ARMNN_NO_DEPRECATE_WARN_BEGIN
    armnn::IConnectableLayer* const startLayer = data.m_Network->AddMultiplicationLayer();
    ARMNN_NO_DEPRECATE_WARN_END
    startLayer->SetBackendId(setBackend);

    bool isReshapeSupported = BroadcastTensor(input0, input1, startLayer, data);
    if (!isReshapeSupported)
    {
        return false;
    }

    return SetupAndTrackLayerOutputSlot(operation, 0, *startLayer, model,
                                        data, nullptr, validateFunc, activationFunction);
}

bool Converter::ConvertPad(const Operation& operation, const Model& model, ConversionData& data)
{
    VLOG(DRIVER) << "Converter::ConvertPad()";

    LayerInputHandle input = ConvertToLayerInputHandle(operation, 0, model, data);
    if (!input.IsValid())
    {
        return Fail("%s: Operation has invalid inputs", __func__);
    }

    const armnn::TensorInfo& inputInfo = input.GetTensorInfo();
    unsigned int rank = inputInfo.GetNumDimensions();

    armnn::PadDescriptor descriptor;
    if (!ConvertPaddings(operation, model, data, rank, descriptor))
    {
        return Fail("%s: Could not convert paddings", __func__);
    }

    // For a ANEURALNETWORKS_TENSOR_QUANT8_ASYMM and ANEURALNETWORKS_TENSOR_QUANT8_ASYMM_SIGNED tensor,
    // the scale and zeroPoint must be the same as input0
    // Before Android Q, the pad value for ANEURALNETWORKS_TENSOR_QUANT8_ASYMM was undefined. Since Android Q the pad
    // value must be "logical zero" we set it to be equal to the QuantizationOffset so effectively it ends up as
    // (QuantizationOffset - QuantizationOffset) * scale = 0.
    if (inputInfo.GetDataType() == armnn::DataType::QAsymmU8 || inputInfo.GetDataType() == armnn::DataType::QAsymmS8)
    {
        descriptor.m_PadValue = inputInfo.GetQuantizationOffset();
    }

    const Operand* output = GetOutputOperand(operation, 0, model);
    if (!output)
    {
        return Fail("%s: Could not read output", __func__);
    }

    const armnn::TensorInfo& outputInfo = GetTensorInfoForOperand(*output);

    bool isSupported = false;
    armnn::BackendId setBackend;
    auto validateFunc = [&](const armnn::TensorInfo& outputInfo, bool& isSupported)
    {
        FORWARD_LAYER_SUPPORT_FUNC(__func__,
                                   IsPadSupported,
                                   data.m_Backends,
                                   isSupported,
                                   setBackend,
                                   inputInfo,
                                   outputInfo,
                                   descriptor);
    };

    if(!IsDynamicTensor(outputInfo))
    {
        validateFunc(outputInfo, isSupported);
    }
    else
    {
        isSupported = AreDynamicTensorsSupported();
    }

    if (!isSupported)
    {
        return false;
    }

    armnn::IConnectableLayer* const layer = data.m_Network->AddPadLayer(descriptor);
    layer->SetBackendId(setBackend);
    assert(layer != nullptr);
    input.Connect(layer->GetInputSlot(0));

    return SetupAndTrackLayerOutputSlot(operation, 0, *layer, model, data, nullptr, validateFunc);
}

bool Converter::ConvertPadV2(const Operation& operation, const Model& model, ConversionData& data)
{
    VLOG(DRIVER) << "Converter::ConvertPadV2()";

    LayerInputHandle input = ConvertToLayerInputHandle(operation, 0, model, data);
    if (!input.IsValid())
    {
        return Fail("%s: Could not read input 0", __func__);
    }

    const Operand* output = GetOutputOperand(operation, 0, model);
    if (!output)
    {
        return Fail("%s: Could not read output", __func__);
    }

    const TensorInfo& inputInfo  = input.GetTensorInfo();
    unsigned int rank = inputInfo.GetNumDimensions();

    PadDescriptor descriptor;
    if (!ConvertPaddings(operation, model, data, rank, descriptor))
    {
        return Fail("%s: Could not convert paddings", __func__);
    }

    const TensorInfo& outputInfo = GetTensorInfoForOperand(*output);

    // Determine type of padding value
    OperandType operandType0;
    OperandType operandType2;

    if (!GetOperandType(operation, 0, model, operandType0) ||
        !GetOperandType(operation, 2, model, operandType2))
    {
        return Fail("%s: Operation has invalid inputs", __func__);
    }

    // Read value to use for padding
    if (operandType0 == OperandType::TENSOR_FLOAT16 && operandType2 == OperandType::FLOAT16)
    {
        Half f16PadValue;
        if (!GetInputScalar(operation, 2, operandType2, f16PadValue, model, data))
        {
            return Fail("%s: Could not read input 2 (FLOAT16)", __func__);
        }

        descriptor.m_PadValue = f16PadValue;
    }
    else if (operandType0 == OperandType::TENSOR_FLOAT32 && operandType2 == OperandType::FLOAT32)
    {
        if (!GetInputFloat32(operation, 2, descriptor.m_PadValue, model, data))
        {
            return Fail("%s: Could not read input 2 (FLOAT32)", __func__);
        }
    }
    else if (isQuantizedOperand(operandType0) && operandType2 == OperandType::INT32)
    {
        int32_t intPadValue = 0;
        if (!GetInputInt32(operation, 2, intPadValue, model, data))
        {
            return Fail("%s: Could not read input 2 (INT32)", __func__);
        }
        descriptor.m_PadValue = intPadValue;
    }
    else
    {
        return Fail("%s: Operation has invalid inputs: type mismatch", __func__);
    }

    bool isSupported = false;
    armnn::BackendId setBackend;
    auto validateFunc = [&](const armnn::TensorInfo& outputInfo, bool& isSupported)
    {
        FORWARD_LAYER_SUPPORT_FUNC(__func__,
                                   IsPadSupported,
                                   data.m_Backends,
                                   isSupported,
                                   setBackend,
                                   inputInfo,
                                   outputInfo,
                                   descriptor);
    };

    if(IsDynamicTensor(outputInfo))
    {
        isSupported = AreDynamicTensorsSupported();
    }
    else
    {
        validateFunc(outputInfo, isSupported);
    }

    if (!isSupported)
    {
        return false;
    }

    IConnectableLayer* const layer = data.m_Network->AddPadLayer(descriptor);
    layer->SetBackendId(setBackend);
    assert(layer != nullptr);
    input.Connect(layer->GetInputSlot(0));

    return SetupAndTrackLayerOutputSlot(operation, 0, *layer, model, data, nullptr, validateFunc);
}

bool Converter::ConvertPrelu(const Operation& operation, const Model& model, ConversionData& data)
{
    VLOG(DRIVER) << "Converter::ConvertPrelu()";

    LayerInputHandle input = ConvertToLayerInputHandle(operation, 0, model, data);
    LayerInputHandle alpha = ConvertToLayerInputHandle(operation, 1, model, data);

    if (!input.IsValid() || !alpha.IsValid())
    {
        return Fail("%s: Operation has invalid inputs", __func__);
    }

    const Operand* output = GetOutputOperand(operation, 0, model);

    if (!output)
    {
        return Fail("%s: Could not read output", __func__);
    }

    const TensorInfo& inputInfo  = input.GetTensorInfo();
    const TensorInfo& alphaInfo  = alpha.GetTensorInfo();
    const TensorInfo& outputInfo = GetTensorInfoForOperand(*output);

    bool isSupported = false;
    armnn::BackendId setBackend;
    auto validateFunc = [&](const armnn::TensorInfo& outputInfo, bool& isSupported)
    {
        FORWARD_LAYER_SUPPORT_FUNC(__func__,
                                   IsPreluSupported,
                                   data.m_Backends,
                                   isSupported,
                                   setBackend,
                                   inputInfo,
                                   alphaInfo,
                                   outputInfo);
    };

    if(IsDynamicTensor(outputInfo))
    {
        isSupported = AreDynamicTensorsSupported();
    }
    else
    {
        validateFunc(outputInfo, isSupported);
    }

    if (!isSupported)
    {
        return false;
    }

    IConnectableLayer* const layer = data.m_Network->AddPreluLayer();
    layer->SetBackendId(setBackend);

    if (!layer)
    {
        return Fail("%s: AddPreluLayer failed", __func__);
    }

    bool isReshapeSupported = BroadcastTensor(input, alpha, layer, data);
    if (!isReshapeSupported)
    {
        return false;
    }

    return SetupAndTrackLayerOutputSlot(operation, 0, *layer, model, data, nullptr, validateFunc);
}

bool Converter::ConvertQuantize(const Operation& operation, const Model& model, ConversionData& data)
{
    VLOG(DRIVER) << "Converter::ConvertQuantize()";

    LayerInputHandle input = ConvertToLayerInputHandle(operation, 0, model, data);
    if (!input.IsValid())
    {
        return Fail("%s: Operation has invalid input", __func__);
    }

    const Operand* const outputOperand = GetOutputOperand(operation, 0, model);
    if (!outputOperand)
    {
        return Fail("%s: Operation has invalid outputs", __func__);
    }

    const TensorInfo& outputInfo = GetTensorInfoForOperand(*outputOperand);

    bool isSupported = false;
    armnn::BackendId setBackend;
    auto validateFunc = [&](const armnn::TensorInfo& outputInfo, bool& isSupported)
    {
        FORWARD_LAYER_SUPPORT_FUNC(__func__,
                                   IsQuantizeSupported,
                                   data.m_Backends,
                                   isSupported,
                                   setBackend,
                                   input.GetTensorInfo(),
                                   outputInfo);
    };

    if(IsDynamicTensor(outputInfo))
    {
        isSupported = AreDynamicTensorsSupported();
    }
    else
    {
        validateFunc(outputInfo, isSupported);
    }

    if (!isSupported)
    {
        return false;
    }

    IConnectableLayer* const layer = data.m_Network->AddQuantizeLayer();
    layer->SetBackendId(setBackend);
    assert(layer != nullptr);
    input.Connect(layer->GetInputSlot(0));

    return SetupAndTrackLayerOutputSlot(operation, 0, *layer, model, data, nullptr, validateFunc);
}

bool Converter::ConvertQuantizedLstm(const Operation& operation, const Model& model, ConversionData& data)
{
    VLOG(DRIVER) << "Converter::ConvertQuantizedLstm()";

    VLOG(DRIVER) << "ConvertQuantizedLstm()";

    //Inputs:
    // 0: The input: A 2-D tensor of type ANEURALNETWORKS_TENSOR_QUANT8_ASYMM and shape [numBatches, inputSize]
    //    specifying the input to the LSTM cell. Tensor is quantized with a fixed quantization range of -1, 127/128.
    LayerInputHandle input = ConvertToLayerInputHandle(operation, 0, model, data);
    if (!input.IsValid())
    {
        return Fail("%s: Could not read input 0: input", __func__);
    }

    // 18: The output state: A 2-D tensor of ANEURALNETWORKS_TENSOR_QUANT8_ASYMM, of shape [batch_size, output_size].
    LayerInputHandle outputStatePrevTimeStep = ConvertToLayerInputHandle(operation, 18, model, data);
    if (!outputStatePrevTimeStep.IsValid())
    {
        return Fail("%s: Could not read input 18: outputStatePrevTimeStep", __func__);
    }

    // 19: The cell state: A 2-D tensor of ANEURALNETWORKS_TENSOR_QUANT16_SYMM, of shape [batch_size, num_units].
    LayerInputHandle cellStatePrevTimeStep = ConvertToLayerInputHandle(operation, 19, model, data);
    if (!cellStatePrevTimeStep.IsValid())
    {
        return Fail("%s: Could not read input 19: cellStatePrevTimeStep", __func__);
    }

    // Get the mandatory input tensors:

    // 02: The input-to-forget weights: A 2-D tensor of ANEURALNETWORKS_TENSOR_QUANT8_SYMM, of shape
    //     [num_units, input_size].
    const ConstTensorPin inputToForgetWeightsPin =
            ConvertOperationInputToConstTensorPin(operation, 2, model, data);

    // 03: The input-to-cell weights: A 2-D tensor of ANEURALNETWORKS_TENSOR_QUANT8_SYMM, of shape
    // [num_units, input_size].
    const ConstTensorPin inputToCellWeightsPin =
            ConvertOperationInputToConstTensorPin(operation, 3, model, data);

    // 04: The input-to-output weights: A 2-D tensor of ANEURALNETWORKS_TENSOR_QUANT8_SYMM, of shape
    //     [num_units, input_size].
    const ConstTensorPin inputToOutputWeightsPin =
            ConvertOperationInputToConstTensorPin(operation, 4, model, data);

    // 06: The recurrent-to-forget weights: A 2-D tensor of ANEURALNETWORKS_TENSOR_QUANT8_SYMM, of shape
    //     [num_units, output_size].
    const ConstTensorPin recurrentToForgetWeightsPin =
            ConvertOperationInputToConstTensorPin(operation, 6, model, data);

    // 07: The recurrent-to-cell weights: A 2-D tensor of ANEURALNETWORKS_TENSOR_QUANT8_SYMM, of shape
    //     [num_units, output_size].
    const ConstTensorPin recurrentToCellWeightsPin =
            ConvertOperationInputToConstTensorPin(operation, 7, model, data);

    // 08: The recurrent-to-output weights: A 2-D tensor of ANEURALNETWORKS_TENSOR_QUANT8_SYMM, of shape
    //     [num_units, output_size].
    const ConstTensorPin recurrentToOutputWeightsPin =
            ConvertOperationInputToConstTensorPin(operation, 8, model, data);

    // 13: The forget gate bias: A 1-D tensor of ANEURALNETWORKS_TENSOR_INT32, of shape [num_units].
    const ConstTensorPin forgetGateBiasPin =
            ConvertOperationInputToConstTensorPin(operation, 13, model, data);

    // 14: The cell bias: A 1-D tensor of ANEURALNETWORKS_TENSOR_INT32, of shape [num_units].
    const ConstTensorPin cellBiasPin =
            ConvertOperationInputToConstTensorPin(operation, 14, model, data);

    // 15: The output gate bias: A 1-D tensor of ANEURALNETWORKS_TENSOR_INT32, of shape [num_units].
    const ConstTensorPin outputGateBiasPin =
            ConvertOperationInputToConstTensorPin(operation, 15, model, data);

    if (!inputToForgetWeightsPin.IsValid() ||
        !inputToCellWeightsPin.IsValid() ||
        !inputToOutputWeightsPin.IsValid() ||
        !recurrentToForgetWeightsPin.IsValid() ||
        !recurrentToCellWeightsPin.IsValid() ||
        !recurrentToOutputWeightsPin.IsValid() ||
        !forgetGateBiasPin.IsValid() ||
        !cellBiasPin.IsValid() ||
        !outputGateBiasPin.IsValid())
    {
        return Fail("%s: Operation has invalid tensor inputs", __func__);
    }

    // Get the optional input tensors:

    // 01: The input-to-input weights: Optional. A 2-D tensor of ANEURALNETWORKS_TENSOR_QUANT8_SYMM, of shape
    //     [num_units, input_size], where “num_units” corresponds to the number of cell units.
    const ConstTensorPin inputToInputWeightsPin =
            ConvertOperationInputToConstTensorPin(operation,
                                                  1,
                                                  model,
                                                  data,
                                                  g_DontPermute,
                                                  nullptr,
                                                  true);

    // 05: The recurrent-to-input weights: Optional. A 2-D tensor of ANEURALNETWORKS_TENSOR_QUANT8_SYMM, of shape
    //     [num_units, output_size], where “output_size” corresponds to either the number of cell units (i.e.,
    //     “num_units”), or the second dimension of the “projection_weights”, if defined.
    const ConstTensorPin recurrentToInputWeightsPin =
            ConvertOperationInputToConstTensorPin(operation,
                                                  5,
                                                  model,
                                                  data,
                                                  g_DontPermute,
                                                  nullptr,
                                                  true);

    // 09: The cell-to-input weights: Optional. A 1-D tensor of ANEURALNETWORKS_TENSOR_QUANT16_SYMM, of shape
    // [num_units].
    const ConstTensorPin cellToInputWeightsPin =
            ConvertOperationInputToConstTensorPin(operation,
                                                  9,
                                                  model,
                                                  data,
                                                  g_DontPermute,
                                                  nullptr,
                                                  true);

    // 10: The cell-to-forget weights: Optional. A 1-D tensor of ANEURALNETWORKS_TENSOR_QUANT16_SYMM, of shape
    // [num_units].
    const ConstTensorPin cellToForgetWeightsPin =
            ConvertOperationInputToConstTensorPin(operation,
                                                  10,
                                                  model,
                                                  data,
                                                  g_DontPermute,
                                                  nullptr,
                                                  true);

    // 11: The cell-to-output weights: Optional. A 1-D tensor of ANEURALNETWORKS_TENSOR_QUANT16_SYMM, of shape
    // [num_units].
    const ConstTensorPin cellToOutputWeightsPin =
            ConvertOperationInputToConstTensorPin(operation,
                                                  11,
                                                  model,
                                                  data,
                                                  g_DontPermute,
                                                  nullptr,
                                                  true);

    // 12: The input gate bias: Optional. A 1-D tensor of ANEURALNETWORKS_TENSOR_INT32, of shape [num_units].
    const ConstTensorPin inputGateBiasPin =
            ConvertOperationInputToConstTensorPin(operation,
                                                  12,
                                                  model,
                                                  data,
                                                  g_DontPermute,
                                                  nullptr,
                                                  true);

    // 16: The projection weights: Optional. A 2-D tensor of ANEURALNETWORKS_TENSOR_QUANT8_SYMM, of shape
    //     [output_size, num_units].
    const ConstTensorPin projectionWeightsPin =
            ConvertOperationInputToConstTensorPin(operation,
                                                  16,
                                                  model,
                                                  data,
                                                  g_DontPermute,
                                                  nullptr,
                                                  true);

    // 17: The projection bias: Optional. A 1-D tensor of ANEURALNETWORKS_TENSOR_INT32, of shape [output_size].
    const ConstTensorPin projectionBiasPin =
            ConvertOperationInputToConstTensorPin(operation,
                                                  17,
                                                  model,
                                                  data,
                                                  g_DontPermute,
                                                  nullptr,
                                                  true);

    if ((!inputToInputWeightsPin.IsValid() && !inputToInputWeightsPin.IsOptional())
        || (!recurrentToInputWeightsPin.IsValid() && !recurrentToInputWeightsPin.IsOptional())
        || (!cellToInputWeightsPin.IsValid() && !cellToInputWeightsPin.IsOptional())
        || (!cellToForgetWeightsPin.IsValid() && !cellToForgetWeightsPin.IsOptional())
        || (!cellToOutputWeightsPin.IsValid() && !cellToOutputWeightsPin.IsOptional())
        || (!inputGateBiasPin.IsValid() && !inputGateBiasPin.IsOptional())
        || (!projectionWeightsPin.IsValid() && !projectionWeightsPin.IsOptional())
        || (!projectionBiasPin.IsValid() && !projectionBiasPin.IsOptional()))
    {
        return Fail("%s: Operation has invalid tensor inputs", __func__);
    }


    // Get the optional normalization tensors

    // 20: The input layer normalization weights. A 1-D tensor of shape [num_units] ANEURALNETWORKS_TENSOR_QUANT16_SYMM.
    //     Used to rescale normalized inputs to activation at input gate.
    const ConstTensorPin inputLayerNormWeightsPin =
            ConvertOperationInputToConstTensorPin(operation,
                                                  20,
                                                  model,
                                                  data,
                                                  g_DontPermute,
                                                  nullptr,
                                                  true);

    // 21: The forget layer normalization weights. A 1-D tensor of shape [num_units] ANEURALNETWORKS_TENSOR_QUANT16_SYMM
    //     Used to rescale normalized inputs to activation at forget gate.
    const ConstTensorPin forgetLayerNormWeightsPin =
            ConvertOperationInputToConstTensorPin(operation,
                                                  21,
                                                  model,
                                                  data,
                                                  g_DontPermute,
                                                  nullptr,
                                                  true);

    // 22: The cell layer normalization weights. A 1-D tensor of shape [num_units] ANEURALNETWORKS_TENSOR_QUANT16_SYMM.
    //     Used to rescale normalized inputs to activation at cell gate.
    const ConstTensorPin cellLayerNormWeightsPin =
            ConvertOperationInputToConstTensorPin(operation,
                                                  22,
                                                  model,
                                                  data,
                                                  g_DontPermute,
                                                  nullptr,
                                                  true);

    // 23: The output layer normalization weights. A 1-D tensor of shape [num_units].
    //     Used to rescale normalized inputs to activation at output gate.
    const ConstTensorPin outputLayerNormWeightsPin =
            ConvertOperationInputToConstTensorPin(operation,
                                                  23,
                                                  model,
                                                  data,
                                                  g_DontPermute,
                                                  nullptr,
                                                  true);

    if ((!inputLayerNormWeightsPin.IsValid() && !inputLayerNormWeightsPin.IsOptional())
        || (!forgetLayerNormWeightsPin.IsValid() && !forgetLayerNormWeightsPin.IsOptional())
        || (!cellLayerNormWeightsPin.IsValid() && !cellLayerNormWeightsPin.IsOptional())
        || (!outputLayerNormWeightsPin.IsValid() && !outputLayerNormWeightsPin.IsOptional()))
    {
        return Fail("%s: Operation has invalid tensor inputs", __func__);
    }

    // Get the optional input scalars:
    // 24: The cell clip:  If provided the cell state is clipped by this value prior to the cell output activation.
    // 25: The projection clip: If provided and projection is enabled, this is used for clipping the projected values.

    // Get the mandatory input scalars:
    // 26: The scale of the intermediate result of matmul, i.e. input to layer normalization, at input gate.
    // 27: The scale of the intermediate result of matmul, i.e. input to layer normalization, at forget gate.
    // 28: The scale of the intermediate result of matmul, i.e. input to layer normalization, at cell gate.
    // 29: The scale of the intermediate result of matmul, i.e. input to layer normalization, at output gate.
    // 30: The zero point of the hidden state, i.e. input to projection.
    // 31: The scale of the hidden state, i.e. input to projection.
    float cellClip, projClip, matMulInputGate, matMulForgetGate, matMulCellGate, matMulOutputGate, projInputScale;
    int projInputZeroPoint;

    if (!GetInputScalar(operation, 24, OperandType::FLOAT32, cellClip, model, data, true) ||
        !GetInputScalar(operation, 25, OperandType::FLOAT32, projClip, model, data, true) ||
        !GetInputScalar(operation, 26, OperandType::FLOAT32, matMulInputGate, model, data) ||
        !GetInputScalar(operation, 27, OperandType::FLOAT32, matMulForgetGate, model, data) ||
        !GetInputScalar(operation, 28, OperandType::FLOAT32, matMulCellGate, model, data) ||
        !GetInputScalar(operation, 29, OperandType::FLOAT32, matMulOutputGate, model, data) ||
        !GetInputScalar(operation, 30, OperandType::INT32, projInputZeroPoint, model, data) ||
        !GetInputScalar(operation, 31, OperandType::FLOAT32, projInputScale, model, data))
    {
        return Fail("%s: Operation has invalid scalar inputs", __func__);
    }

    // Outputs:
    // 0: The output state (out): A 2-D tensor of ANEURALNETWORKS_TENSOR_QUANT8_ASYMM_SIGNED, of shape [batch_size,
    // output_size].
    const Operand* outputStateOut = GetOutputOperand(operation, 0, model);
    if (!outputStateOut)
    {
        return Fail("%s: Could not read output 0: outputStateOut", __func__);
    }

    // 1: The cell state (out): A 2-D tensor of ANEURALNETWORKS_TENSOR_QUANT16_SYMM, of shape [batch_size, num_units].
    const Operand* cellStateOut = GetOutputOperand(operation, 1, model);
    if (!cellStateOut)
    {
        return Fail("%s: Could not read output 1: cellStateOut", __func__);
    }

    // 2: The output: A 2-D tensor of ANEURALNETWORKS_TENSOR_QUANT8_ASYMM_SIGNED, of shape [batch_size, output_size].
    // This is effectively the same as the current “output state (out)” value.
    const Operand* output = GetOutputOperand(operation, 2, model);
    if (!output)
    {
        return Fail("%s: Could not read output 2: output", __func__);
    }

    // set the params structure for the AddLstmLayer call
    LstmInputParams params;
    params.m_InputToInputWeights = inputToInputWeightsPin.GetConstTensorPtr();
    params.m_InputToForgetWeights = inputToForgetWeightsPin.GetConstTensorPtr();
    params.m_InputToCellWeights = inputToCellWeightsPin.GetConstTensorPtr();
    params.m_InputToOutputWeights = inputToOutputWeightsPin.GetConstTensorPtr();
    params.m_RecurrentToInputWeights = recurrentToInputWeightsPin.GetConstTensorPtr();
    params.m_RecurrentToForgetWeights = recurrentToForgetWeightsPin.GetConstTensorPtr();
    params.m_RecurrentToCellWeights = recurrentToCellWeightsPin.GetConstTensorPtr();
    params.m_RecurrentToOutputWeights = recurrentToOutputWeightsPin.GetConstTensorPtr();
    params.m_CellToInputWeights = cellToInputWeightsPin.GetConstTensorPtr();
    params.m_CellToForgetWeights = cellToForgetWeightsPin.GetConstTensorPtr();
    params.m_CellToOutputWeights = cellToOutputWeightsPin.GetConstTensorPtr();
    params.m_InputGateBias = inputGateBiasPin.GetConstTensorPtr();
    params.m_ForgetGateBias = forgetGateBiasPin.GetConstTensorPtr();
    params.m_CellBias = cellBiasPin.GetConstTensorPtr();
    params.m_OutputGateBias = outputGateBiasPin.GetConstTensorPtr();
    params.m_ProjectionWeights = projectionWeightsPin.GetConstTensorPtr();
    params.m_ProjectionBias = projectionBiasPin.GetConstTensorPtr();
    params.m_InputLayerNormWeights = inputLayerNormWeightsPin.GetConstTensorPtr();
    params.m_ForgetLayerNormWeights = forgetLayerNormWeightsPin.GetConstTensorPtr();
    params.m_CellLayerNormWeights = cellLayerNormWeightsPin.GetConstTensorPtr();
    params.m_OutputLayerNormWeights = outputLayerNormWeightsPin.GetConstTensorPtr();

    // set the layer descriptor
    QLstmDescriptor desc;
    desc.m_CellClip = cellClip;
    desc.m_ProjectionClip = projClip;
    desc.m_CifgEnabled = (params.m_InputToInputWeights == nullptr ||
                          params.m_RecurrentToInputWeights == nullptr ||
                          params.m_InputGateBias == nullptr);
    desc.m_PeepholeEnabled = (params.m_CellToForgetWeights != nullptr ||
                              params.m_CellToOutputWeights != nullptr);
    desc.m_ProjectionEnabled = (params.m_ProjectionWeights != nullptr);
    desc.m_LayerNormEnabled = (params.m_InputLayerNormWeights != nullptr ||
                               params.m_ForgetLayerNormWeights != nullptr ||
                               params.m_CellLayerNormWeights != nullptr ||
                               params.m_OutputLayerNormWeights != nullptr);
    desc.m_InputIntermediateScale = matMulInputGate;
    desc.m_ForgetIntermediateScale = matMulForgetGate;
    desc.m_CellIntermediateScale = matMulCellGate;
    desc.m_OutputIntermediateScale = matMulOutputGate;
    desc.m_HiddenStateScale = projInputScale;
    desc.m_HiddenStateZeroPoint = projInputZeroPoint;

    // validate the optional input groups
    if (desc.m_CifgEnabled &&
        (params.m_InputToInputWeights != nullptr ||
         params.m_RecurrentToInputWeights != nullptr ||
         params.m_InputGateBias != nullptr))
    {
        return Fail("%s: All, or none, of input-to-input weights, recurrent-to-input weights,"
                    " and input gate bias must be provided", __func__);
    }

    if (!desc.m_ProjectionEnabled && params.m_ProjectionBias != nullptr)
    {
        return Fail("%s: projection bias should not be provided without projection weights", __func__);
    }

    if (desc.m_PeepholeEnabled &&
        (params.m_CellToForgetWeights == nullptr ||
         params.m_CellToOutputWeights == nullptr ||
         (!desc.m_CifgEnabled && params.m_CellToInputWeights == nullptr)))
    {
        return Fail("%s: All, or none, of cell-to-forget weights and cell-to-output weights must be provided"
                    " and, if CIFG is not enabled, cell-to-input weights must also be provided", __func__);
    }

    if (desc.m_LayerNormEnabled &&
        (params.m_ForgetLayerNormWeights == nullptr ||
         params.m_CellLayerNormWeights == nullptr ||
         params.m_OutputLayerNormWeights == nullptr ||
         (!desc.m_CifgEnabled && params.m_InputLayerNormWeights == nullptr)))
    {
        return Fail("%s: All, or none, of forget-norm weights, cell-norm weights and output-norm weights must be"
                    " provided and, if CIFG is not enabled, input-norm weights must also be provided", __func__);
    }

    // Basic parameters
    LstmInputParamsInfo paramsInfo;
    paramsInfo.m_InputToForgetWeights     = &(params.m_InputToForgetWeights->GetInfo());
    paramsInfo.m_InputToCellWeights       = &(params.m_InputToCellWeights->GetInfo());
    paramsInfo.m_InputToOutputWeights     = &(params.m_InputToOutputWeights->GetInfo());
    paramsInfo.m_RecurrentToForgetWeights = &(params.m_RecurrentToForgetWeights->GetInfo());
    paramsInfo.m_RecurrentToCellWeights   = &(params.m_RecurrentToCellWeights->GetInfo());
    paramsInfo.m_RecurrentToOutputWeights = &(params.m_RecurrentToOutputWeights->GetInfo());
    paramsInfo.m_ForgetGateBias           = &(params.m_ForgetGateBias->GetInfo());
    paramsInfo.m_CellBias                 = &(params.m_CellBias->GetInfo());
    paramsInfo.m_OutputGateBias           = &(params.m_OutputGateBias->GetInfo());

    // Inputs
    const TensorInfo& inputInfo = input.GetTensorInfo();
    const TensorInfo& outputStatePrevTimeStepInfo = outputStatePrevTimeStep.GetTensorInfo();
    const TensorInfo& cellStatePrevTimeStepInfo = cellStatePrevTimeStep.GetTensorInfo();

    // Outputs
    TensorInfo outputStateOutInfo = GetTensorInfoForOperand(*outputStateOut);
    TensorInfo outputInfo = GetTensorInfoForOperand(*output);
    const TensorInfo& cellStateOutInfo = GetTensorInfoForOperand(*cellStateOut);

    // Optional parameters
    if (!desc.m_CifgEnabled)
    {
        paramsInfo.m_InputToInputWeights = &(params.m_InputToInputWeights->GetInfo());
        paramsInfo.m_RecurrentToInputWeights = &(params.m_RecurrentToInputWeights->GetInfo());
        if (desc.m_PeepholeEnabled)
        {
            paramsInfo.m_CellToInputWeights = &(params.m_CellToInputWeights->GetInfo());
        }
        paramsInfo.m_InputGateBias = &(params.m_InputGateBias->GetInfo());
    }


    if (desc.m_ProjectionEnabled)
    {
        paramsInfo.m_ProjectionWeights = &(params.m_ProjectionWeights->GetInfo());
        if (params.m_ProjectionBias != nullptr)
        {
            paramsInfo.m_ProjectionBias = &(params.m_ProjectionBias->GetInfo());
        }
    }
    else
    {
        // If Projection is disabled, override non-const outputs to change the quant info with hidden params, then
        // create a new const TensorInfo based on this
        outputStateOutInfo.SetQuantizationScale(projInputScale);
        outputStateOutInfo.SetQuantizationOffset(projInputZeroPoint);
        outputInfo.SetQuantizationScale(projInputScale);
        outputInfo.SetQuantizationOffset(projInputZeroPoint);
    }

    const TensorInfo constOutputStateOutInfo(outputStateOutInfo);
    const TensorInfo constOutputInfo(outputInfo);

    if (desc.m_PeepholeEnabled)
    {
        paramsInfo.m_CellToForgetWeights = &(params.m_CellToForgetWeights->GetInfo());
        paramsInfo.m_CellToOutputWeights = &(params.m_CellToOutputWeights->GetInfo());
    }

    if (desc.m_LayerNormEnabled)
    {
        if(!desc.m_CifgEnabled)
        {
            paramsInfo.m_InputLayerNormWeights = &(params.m_InputLayerNormWeights->GetInfo());
        }
        paramsInfo.m_ForgetLayerNormWeights = &(params.m_ForgetLayerNormWeights->GetInfo());
        paramsInfo.m_CellLayerNormWeights = &(params.m_CellLayerNormWeights->GetInfo());
        paramsInfo.m_OutputLayerNormWeights = &(params.m_OutputLayerNormWeights->GetInfo());
    }

    // Check if the layer is supported
    bool isSupported = false;
    armnn::BackendId setBackend;
    auto validateFunc = [&](const armnn::TensorInfo& cellStateOutInfo, bool& isSupported)
    {
        FORWARD_LAYER_SUPPORT_FUNC(__func__,
                                   IsQLstmSupported,
                                   data.m_Backends,
                                   isSupported,
                                   setBackend,
                                   inputInfo,
                                   outputStatePrevTimeStepInfo,
                                   cellStatePrevTimeStepInfo,
                                   constOutputStateOutInfo,
                                   cellStateOutInfo,
                                   constOutputInfo,
                                   desc,
                                   paramsInfo);
    };

    bool isDynamic = false;
    if (!IsDynamicTensor(constOutputStateOutInfo) &&
        !IsDynamicTensor(cellStateOutInfo)  &&
        !IsDynamicTensor(constOutputInfo))
    {
        validateFunc(outputInfo, isSupported);
    }
    else
    {
        isDynamic = true;
        isSupported = AreDynamicTensorsSupported();
    }

    if (!isSupported)
    {
        return false;
    }

    // Add the layer
    IConnectableLayer* layer = data.m_Network->AddQLstmLayer(desc, params, "QLstm");
    layer->SetBackendId(setBackend);

    input.Connect(layer->GetInputSlot(0));
    outputStatePrevTimeStep.Connect(layer->GetInputSlot(1));
    cellStatePrevTimeStep.Connect(layer->GetInputSlot(2));

    if (!isDynamic)
    {
        return ( SetupAndTrackLayerOutputSlot(
                operation, 0, *layer, 0, model, data, &constOutputStateOutInfo) &&
                 SetupAndTrackLayerOutputSlot(operation, 1, *layer, 1, model, data) &&
                 SetupAndTrackLayerOutputSlot(operation, 2, *layer, 2, model, data, &constOutputInfo));
    }
    else
    {
        return ( SetupAndTrackLayerOutputSlot(
                operation, 0, *layer, 0, model, data, &constOutputStateOutInfo) &&
                 SetupAndTrackLayerOutputSlot(
                         operation, 1, *layer, 1, model, data, nullptr, validateFunc,
                         ActivationFn::kActivationNone, true) &&
                 SetupAndTrackLayerOutputSlot(operation, 2, *layer, 2, model, data, &constOutputInfo));
    }
}

bool Converter::ConvertQuantized16BitLstm(const Operation& operation, const Model& model, ConversionData& data)
{
    VLOG(DRIVER) << "Converter::ConvertQuantized16BitLstm()";
    VLOG(DRIVER) << "Policy::ConvertQuantized16BitLstm()";

    //Inputs:
    // 0: The input: A 2-D tensor of type ANEURALNETWORKS_TENSOR_QUANT8_ASYMM and shape [numBatches, inputSize]
    //    specifying the input to the LSTM cell. Tensor is quantized with a fixed quantization range of -1, 127/128.
    LayerInputHandle input = ConvertToLayerInputHandle(operation, 0, model, data);
    if (!input.IsValid())
    {
        return Fail("%s: Could not read input 0: input", __func__);
    }

    //13: The previous cell state: A 2-D tensor of type ANEURALNETWORKS_TENSOR_QUANT16_SYMM and shape
    //    [numBatches, outputSize] specifying the cell state from the previous time step of the LSTM cell.
    //    It is quantized using a quantization range of -2^4, 2^4 * 32767/32768.
    LayerInputHandle previousCellStateIn = ConvertToLayerInputHandle(operation, 13, model, data);
    if (!previousCellStateIn.IsValid())
    {
        return Fail("%s: Could not read input 13: previousCellStateIn", __func__);
    }

    // 14: The previous output state: A 2-D tensor of type ANEURALNETWORKS_TENSOR_QUANT8_ASYMM and shape
    //     [numBathes, outputSize] specifying the output of the LSTM cell from previous time-step. Tensor
    //     is quantized with a fixed quantization range of -1, 127/128.
    LayerInputHandle previousOutputIn = ConvertToLayerInputHandle(operation, 14, model, data);
    if (!previousOutputIn.IsValid())
    {
        return Fail("%s: Could not read input 14: previousOutputIn", __func__);
    }

    // Get the input tensors:
    // 1: The input-to-input weights. A 2-D tensor of type ANEURALNETWORKS_TENSOR_QUANT8_ASYMM and shape
    //    [outputSize, inputSize] specifying input-to-input part of weights for fully-connected layer inside the
    //    LSTM cell. Quantization zero point and scale must be the same across all the weights.
    const ConstTensorPin inputToInputWeightsPin =
        ConvertOperationInputToConstTensorPin(operation, 1, model, data);

    // 2: The input-to-forget weights. A 2-D tensor of type ANEURALNETWORKS_TENSOR_QUANT8_ASYMM and shape
    //    [outputSize, inputSize] specifying input-to-forget part of weights for fully-connected layer inside the
    //    LSTM cell. Quantization zero point and scale must be the same across all the weights.
    const ConstTensorPin inputToForgetWeightsPin =
        ConvertOperationInputToConstTensorPin(operation, 2, model, data);

    // 3: The input-to-cell weights. A 2-D tensor of type ANEURALNETWORKS_TENSOR_QUANT8_ASYMM and shape
    //    [outputSize, inputSize] specifying input-to-cell part of weights for fully-connected layer inside the
    //    LSTM cell. Quantization zero point and scale must be the same across all the weights.
    const ConstTensorPin inputToCellWeightsPin =
        ConvertOperationInputToConstTensorPin(operation, 3, model, data);

    // 4: The input-to-output weights. A 2-D tensor of type ANEURALNETWORKS_TENSOR_QUANT8_ASYMM and shape
    //    [outputSize, inputSize] specifying input-to-output part of weights for fully-connected layer inside the
    //    LSTM cell. Quantization zero point and scale must be the same across all the weights.
    const ConstTensorPin inputToOutputWeightsPin =
        ConvertOperationInputToConstTensorPin(operation, 4, model, data);

    // 5: The recurrent-to-input weights. A 2-D tensor of type ANEURALNETWORKS_TENSOR_QUANT8_ASYMM and shape
    //    [outputSize, outputSize] specifying recurrent-to-input part of weights for fully-connected layer inside
    //    the LSTM cell. Quantization zero point and scale must be the same across all the weights.
    const ConstTensorPin recurrentToInputWeightsPin =
        ConvertOperationInputToConstTensorPin(operation, 5, model, data);

    // 6: The recurrent-to-forget weights. A 2-D tensor of type ANEURALNETWORKS_TENSOR_QUANT8_ASYMM and shape
    //    [outputSize, outputSize] specifying recurrent-to-forget part of weights for fully-connected layer inside
    //    the LSTM cell. Quantization zero point and scale must be the same across all the weights.
    const ConstTensorPin recurrentToForgetWeightsPin =
        ConvertOperationInputToConstTensorPin(operation, 6, model, data);

    // 7: The recurrent-to-cell weights. A 2-D tensor of type ANEURALNETWORKS_TENSOR_QUANT8_ASYMM and shape
    //    [outputSize, outputSize] specifying recurrent-to-cell part of weights for fully-connected layer inside
    //    the LSTM cell. Quantization zero point and scale must be the same across all the weights.
    const ConstTensorPin recurrentToCellWeightsPin =
        ConvertOperationInputToConstTensorPin(operation, 7, model, data);

    // 8: The recurrent-to-output weights. A 2-D tensor of type ANEURALNETWORKS_TENSOR_QUANT8_ASYMM and shape
    //    [outputSize, outputSize] specifying recurrent-to-output part of weights for fully-connected layer inside
    //    the LSTM cell. Quantization zero point and scale must be the same across all the weights.
    const ConstTensorPin recurrentToOutputWeightsPin =
        ConvertOperationInputToConstTensorPin(operation, 8, model, data);

    // 9: The input gate bias. A 1-D tensor of type ANEURALNETWORKS_TENSOR_INT32 and shape [outputSize] specifying the
    //    bias for the fully-connected layer inside the LSTM cell. Bias is quantized with scale being a product
    //    of input and weights scales and zeroPoint equal to 0.
    const ConstTensorPin inputGateBiasPin =
        ConvertOperationInputToConstTensorPin(operation, 9, model, data);

    // 10: The forget gate bias. A 1-D tensor of type ANEURALNETWORKS_TENSOR_INT32 and shape [outputSize] specifying
    //     the bias for the fully-connected layer inside the LSTM cell. Bias is quantized with scale being a product
    //     of input and weights scales and zeroPoint equal to 0.
    const ConstTensorPin forgetGateBiasPin =
        ConvertOperationInputToConstTensorPin(operation, 10, model, data);

    // 11:The cell bias. A 1-D tensor of type ANEURALNETWORKS_TENSOR_INT32 and shape [outputSize] specifying the bias
    //    for the fully-connected layer inside the LSTM cell. Bias is quantized with scale being a product of input
    //    and weights scales and zeroPoint equal to 0.
    const ConstTensorPin cellBiasPin =
        ConvertOperationInputToConstTensorPin(operation, 11, model, data);

    // 12:The output gate bias. A 1-D tensor of type ANEURALNETWORKS_TENSOR_INT32 and shape [outputSize] specifying
    //    the bias for the fully-connected layer inside the LSTM cell. Bias is quantized with scale being a product
    //    of input and weights scales and zeroPoint equal to 0.
    const ConstTensorPin outputGateBiasPin =
        ConvertOperationInputToConstTensorPin(operation, 12, model, data);

    if (!inputToInputWeightsPin.IsValid() ||
        !inputToForgetWeightsPin.IsValid() ||
        !inputToCellWeightsPin.IsValid() ||
        !inputToOutputWeightsPin.IsValid() ||
        !recurrentToInputWeightsPin.IsValid() ||
        !recurrentToForgetWeightsPin.IsValid() ||
        !recurrentToCellWeightsPin.IsValid() ||
        !recurrentToOutputWeightsPin.IsValid() ||
        !inputGateBiasPin.IsValid() ||
        !forgetGateBiasPin.IsValid() ||
        !cellBiasPin.IsValid() ||
        !outputGateBiasPin.IsValid())
    {
        return Fail("%s: Operation has invalid tensor inputs", __func__);
    }

    // Outputs:
    // 0: The cell state: A 2-D tensor of type ANEURALNETWORKS_TENSOR_QUANT16_SYMM and shape [numBatches, outputSize]
    //    which contains a cell state from the current time step. Tensor is quantized using a quantization range
    //    of -2^4, 2^4 * 32767/32768.
    const Operand* cellStateOut = GetOutputOperand(operation, 0, model);
    if (!cellStateOut)
    {
        return Fail("%s: Could not read output 0: cellStateOut", __func__);
    }

    // 1: The output: A 2-D tensor of type ANEURALNETWORKS_TENSOR_QUANT8_ASYMM and shape [numBathes, outputSize] which
    //      contains the output value. Tensor is quantized with a fixed quantization range of -1, 127/128.
    const Operand* output = GetOutputOperand(operation, 1, model);
    if (!output)
    {
        return Fail("%s: Could not read output 1: output", __func__);
    }

    // Inputs
    const TensorInfo& inputInfo               = input.GetTensorInfo();
    const TensorInfo& previousCellStateInInfo = previousCellStateIn.GetTensorInfo();
    const TensorInfo& previousOutputInInfo    = previousOutputIn.GetTensorInfo();

    // Outputs
    const TensorInfo& cellStateOutInfo = GetTensorInfoForOperand(*cellStateOut);
    const TensorInfo& outputInfo       = GetTensorInfoForOperand(*output);

    // Dynamic tensors currently not supported
    if (IsDynamicTensor(cellStateOutInfo) || IsDynamicTensor(outputInfo))
    {
        return Fail("%s: Dynamic output tensors are not supported", __func__);
    }

    QuantizedLstmInputParams params;

    params.m_InputToInputWeights      = inputToInputWeightsPin.GetConstTensorPtr();
    params.m_InputToForgetWeights     = inputToForgetWeightsPin.GetConstTensorPtr();
    params.m_InputToCellWeights       = inputToCellWeightsPin.GetConstTensorPtr();
    params.m_InputToOutputWeights     = inputToOutputWeightsPin.GetConstTensorPtr();
    params.m_RecurrentToInputWeights  = recurrentToInputWeightsPin.GetConstTensorPtr();
    params.m_RecurrentToForgetWeights = recurrentToForgetWeightsPin.GetConstTensorPtr();
    params.m_RecurrentToCellWeights   = recurrentToCellWeightsPin.GetConstTensorPtr();
    params.m_RecurrentToOutputWeights = recurrentToOutputWeightsPin.GetConstTensorPtr();
    params.m_InputGateBias            = inputGateBiasPin.GetConstTensorPtr();
    params.m_ForgetGateBias           = forgetGateBiasPin.GetConstTensorPtr();
    params.m_CellBias                 = cellBiasPin.GetConstTensorPtr();
    params.m_OutputGateBias           = outputGateBiasPin.GetConstTensorPtr();

    QuantizedLstmInputParamsInfo paramsInfo;
    paramsInfo.m_InputToInputWeights      = &(params.m_InputToInputWeights->GetInfo());
    paramsInfo.m_InputToForgetWeights     = &(params.m_InputToForgetWeights->GetInfo());
    paramsInfo.m_InputToCellWeights       = &(params.m_InputToCellWeights->GetInfo());
    paramsInfo.m_InputToOutputWeights     = &(params.m_InputToOutputWeights->GetInfo());
    paramsInfo.m_RecurrentToInputWeights  = &(params.m_RecurrentToInputWeights->GetInfo());
    paramsInfo.m_RecurrentToForgetWeights = &(params.m_RecurrentToForgetWeights->GetInfo());
    paramsInfo.m_RecurrentToCellWeights   = &(params.m_RecurrentToCellWeights->GetInfo());
    paramsInfo.m_RecurrentToOutputWeights = &(params.m_RecurrentToOutputWeights->GetInfo());
    paramsInfo.m_InputGateBias            = &(params.m_InputGateBias->GetInfo());
    paramsInfo.m_ForgetGateBias           = &(params.m_ForgetGateBias->GetInfo());
    paramsInfo.m_CellBias                 = &(params.m_CellBias->GetInfo());
    paramsInfo.m_OutputGateBias           = &(params.m_OutputGateBias->GetInfo());

    bool isSupported = false;
    armnn::BackendId setBackend;
    auto validateFunc = [&](const armnn::TensorInfo& outputInfo, bool& isSupported)
    {
        FORWARD_LAYER_SUPPORT_FUNC(__func__,
                                   IsQuantizedLstmSupported,
                                   data.m_Backends,
                                   isSupported,
                                   setBackend,
                                   inputInfo,
                                   previousCellStateInInfo,
                                   previousOutputInInfo,
                                   cellStateOutInfo,
                                   outputInfo,
                                   paramsInfo);
    };

    bool isDynamic = false;
    if (!IsDynamicTensor(cellStateOutInfo) &&
        !IsDynamicTensor(outputInfo))
    {
        validateFunc(outputInfo, isSupported);
    }
    else
    {
        isDynamic = true;
        isSupported = AreDynamicTensorsSupported();
    }

    if (!isSupported)
    {
        return false;
    }

    IConnectableLayer* const layer = data.m_Network->AddQuantizedLstmLayer(params, "QuantizedLstm");
    layer->SetBackendId(setBackend);
    input.Connect(layer->GetInputSlot(0));
    previousCellStateIn.Connect(layer->GetInputSlot(1));
    previousOutputIn.Connect(layer->GetInputSlot(2));

    if (!isDynamic)
    {
        return (SetupAndTrackLayerOutputSlot(operation, 0, *layer, 0, model, data) &&
                SetupAndTrackLayerOutputSlot(operation, 1, *layer, 1, model, data));
    }
    else
    {
        return (SetupAndTrackLayerOutputSlot(operation, 0, *layer, 0, model, data) &&
                SetupAndTrackLayerOutputSlot(
                    operation, 1, *layer, 1, model, data, nullptr, validateFunc, ActivationFn::kActivationNone, true));
    }

}

bool Converter::ConvertRank(const Operation& operation, const Model& model, ConversionData& data)
{
    VLOG(DRIVER) << "Converter::ConvertRank()";

    const Operand* inputOperand = GetInputOperand(operation, 0, model);
    const Operand* outputOperand = GetOutputOperand(operation, 0, model);

    if (inputOperand == nullptr || outputOperand == nullptr)
    {
        return Fail("%s: Operation has invalid inputs", __func__);
    }

    const Shape inputOperandShape = GetOperandShape(*inputOperand);
    const Shape outputOperandShape = GetOperandShape(*outputOperand);

    LayerInputHandle input = ConvertToLayerInputHandle(operation, 0, model, data);
    if (!input.IsValid())
    {
        return Fail("%s: Could not read input 0", __func__);
    }

    armnn::TensorInfo outInfo = GetTensorInfoForOperand(*outputOperand);
    if (IsDynamicTensor(outInfo))
    {
        return Fail("%s: Dynamic output tensors are not supported", __func__);
    }

    bool isSupported = false;
    armnn::BackendId setBackend;
    FORWARD_LAYER_SUPPORT_FUNC(__func__,
                               IsRankSupported,
                               data.m_Backends,
                               isSupported,
                               setBackend,
                               input.GetTensorInfo(),
                               outInfo);
    if (!isSupported)
    {
        return false;
    }

    armnn::IConnectableLayer* layer = data.m_Network->AddRankLayer();
    layer->SetBackendId(setBackend);
    assert(layer != nullptr);
    input.Connect(layer->GetInputSlot(0));

    return SetupAndTrackLayerOutputSlot(operation, 0, *layer, model, data, &outInfo);
}

bool Converter::ConvertReLu(const Operation& operation, const Model& model, ConversionData& data)
{
    VLOG(DRIVER) << "Converter::ConvertReLu()";
    armnn::ActivationDescriptor desc;
    desc.m_Function = armnn::ActivationFunction::ReLu;


    LayerInputHandle input = ConvertToLayerInputHandle(operation, 0, model, data);
    if (!input.IsValid())
    {
        return Fail("%s: Input 0 is invalid", "operationName");
    }

    const Operand* outputOperand = GetOutputOperand(operation, 0, model);
    if (!outputOperand)
    {
        return false;
    }

    const armnn::TensorInfo& outInfo = GetTensorInfoForOperand(*outputOperand);

    bool isSupported = false;
    armnn::BackendId setBackend;
    auto validateFunc = [&](const armnn::TensorInfo& outInfo, bool& isSupported)
    {
        FORWARD_LAYER_SUPPORT_FUNC(__func__,
                                   IsActivationSupported,
                                   data.m_Backends,
                                   isSupported,
                                   setBackend,
                                   input.GetTensorInfo(),
                                   outInfo,
                                   desc);
    };

    if(IsDynamicTensor(outInfo))
    {
        isSupported = AreDynamicTensorsSupported();
    }
    else
    {
        validateFunc(outInfo, isSupported);
    }

    if (!isSupported)
    {
        return false;
    }

    armnn::IConnectableLayer* layer = data.m_Network->AddActivationLayer(desc);
    layer->SetBackendId(setBackend);
    ARMNN_ASSERT(layer != nullptr);
    input.Connect(layer->GetInputSlot(0));

    return SetupAndTrackLayerOutputSlot(operation, 0, *layer, model, data, nullptr, validateFunc);
}

bool Converter::ConvertReLu1(const Operation& operation, const Model& model, ConversionData& data)
{
    VLOG(DRIVER) << "Converter::ConvertReLu1()";
    armnn::ActivationDescriptor desc;
    desc.m_Function = armnn::ActivationFunction::BoundedReLu;
    desc.m_A        = 1.0f;
    desc.m_B        = -1.0f;

    return ConvertToActivation(operation, __func__, desc, model, data);
}

bool Converter::ConvertReLu6(const Operation& operation, const Model& model, ConversionData& data)
{
    VLOG(DRIVER) << "Converter::ConvertReLu6()";
    armnn::ActivationDescriptor desc;
    desc.m_Function = armnn::ActivationFunction::BoundedReLu;
    desc.m_A        = 6.0f;

    return ConvertToActivation(operation, __func__, desc, model, data);
}

bool Converter::ConvertReshape(const Operation& operation, const Model& model, ConversionData& data)
{
    VLOG(DRIVER) << "Converter::ConvertReshape()";

    const Operand* inputOperand = GetInputOperand(operation, 0, model);
    const Operand* requestedShapeOperand = GetInputOperand(operation, 1, model);
    const Operand* outputOperand = GetOutputOperand(operation, 0, model);

    if (inputOperand == nullptr
        || requestedShapeOperand == nullptr
        || outputOperand == nullptr)
    {
        return Fail("%s: Operation has invalid inputs", __func__);
    }

    if (requestedShapeOperand->dimensions.size() != 1)
    {
        return Fail("%s: Input 1 expected to be one-dimensional (found %i dimensions)",
                    __func__, requestedShapeOperand->dimensions.size());
    }

    std::vector<int32_t> targetDimensions;
    if (!GetTensorInt32Values(*requestedShapeOperand, targetDimensions, model, data))
    {
        return Fail("%s: Could not read values of input 1", __func__);
    }

    const Shape inputOperandShape = GetOperandShape(*inputOperand);

    Shape requestedShape;
    // targetDimensions may contain special values (e.g. -1). reshapePrepare() is an AndroidNN provided utility
    // function that resolves these values into a fully specified tensor shape.
    if (!reshapePrepare(inputOperandShape, targetDimensions.data(), targetDimensions.size(), &requestedShape))
    {
        return Fail("%s: Failed to resolve the requested shape", __func__);
    }

    LayerInputHandle input = ConvertToLayerInputHandle(operation, 0, model, data);
    if (!input.IsValid())
    {
        return Fail("%s: Could not read input 0", __func__);
    }

    armnn::ReshapeDescriptor reshapeDescriptor;
    reshapeDescriptor.m_TargetShape = armnn::TensorShape(requestedShape.dimensions.size(),
                                                         requestedShape.dimensions.data());

    const armnn::TensorInfo& outputInfo = GetTensorInfoForOperand(*outputOperand);

    bool isSupported = false;
    armnn::BackendId setBackend;
    auto validateFunc = [&](const armnn::TensorInfo& outputInfo, bool& isSupported)
    {
        FORWARD_LAYER_SUPPORT_FUNC(__func__,
                                   IsReshapeSupported,
                                   data.m_Backends,
                                   isSupported,
                                   setBackend,
                                   input.GetTensorInfo(),
                                   outputInfo,
                                   reshapeDescriptor);
    };

    if(!IsDynamicTensor(outputInfo))
    {
        validateFunc(outputInfo, isSupported);
    }
    else
    {
        isSupported = AreDynamicTensorsSupported();
    }

    if (!isSupported)
    {
        return false;
    }

    armnn::IConnectableLayer* layer = data.m_Network->AddReshapeLayer(reshapeDescriptor);
    layer->SetBackendId(setBackend);
    assert(layer != nullptr);
    input.Connect(layer->GetInputSlot(0));

    return SetupAndTrackLayerOutputSlot(operation, 0, *layer, model, data, nullptr, validateFunc);
}

bool Converter::ConvertResize(const Operation& operation,
                              const Model& model,
                              ConversionData& data,
                              ResizeMethod resizeMethod)
{
    VLOG(DRIVER) << "Converter::ConvertResize()";
    VLOG(DRIVER) << "resizeMethod = " << GetResizeMethodAsCString(resizeMethod);

    LayerInputHandle input = ConvertToLayerInputHandle(operation, 0, model, data);
    if (!input.IsValid())
    {
        return Fail("%s: Could not read input 0", __func__);
    }

    const Operand* output = GetOutputOperand(operation, 0, model);
    if (!output)
    {
        return Fail("%s: Could not read output 0", __func__);
    }

    const TensorInfo& inputInfo  = input.GetTensorInfo();
    const TensorInfo& outputInfo = GetTensorInfoForOperand(*output);

    ResizeDescriptor descriptor;
    descriptor.m_Method     = resizeMethod;
    descriptor.m_DataLayout = OptionalDataLayout(operation, 3, model, data);

    OperandType operandType1;
    OperandType operandType2;

    if (!GetOperandType(operation, 1, model, operandType1) ||
        !GetOperandType(operation, 2, model, operandType2))
    {
        return Fail("%s: Operation has invalid inputs", __func__);
    }

    if (operandType1 != operandType2)
    {
        return Fail("%s: Operation has invalid inputs. Type of input 1 and 2 should be the same", __func__);
    }

    if (operandType1 == OperandType::INT32)
    {
        // Case 1: resizing by shape
        int32_t targetWidth  = 0;
        int32_t targetHeight = 0;

        if (!GetInputInt32(operation, 1, targetWidth, model, data) ||
            !GetInputInt32(operation, 2, targetHeight, model, data))
        {
            return Fail("%s: Operation has invalid inputs for resizing by shape", __func__);
        }

        if (targetWidth < 0 || targetHeight < 0)
        {
            return Fail("%s: Operation has invalid inputs for resizing by shape. "
                        "Target width/height cannot be < 0", __func__);
        }

        descriptor.m_TargetWidth = static_cast<uint32_t>(targetWidth);
        descriptor.m_TargetHeight = static_cast<uint32_t>(targetHeight);
    }
    else if (operandType1 == OperandType::FLOAT32)
    {
        // Case 2: resizing by scale
        float widthScale  = 1.0f;
        float heightScale = 1.0f;

        if (!GetInputFloat32(operation, 1, widthScale, model, data) ||
            !GetInputFloat32(operation, 2, heightScale, model, data))
        {
            return Fail("%s: Operation has invalid inputs for resizing by scale", __func__);
        }

        const TensorShape& inputShape = inputInfo.GetShape();
        armnnUtils::DataLayoutIndexed dataLayoutIndexed(descriptor.m_DataLayout);

        float width  = inputShape[dataLayoutIndexed.GetWidthIndex()];
        float height = inputShape[dataLayoutIndexed.GetHeightIndex()];

        descriptor.m_TargetWidth  = std::floor(width  * widthScale);
        descriptor.m_TargetHeight = std::floor(height * heightScale);
    }
    else if (operandType1 == OperandType::FLOAT16)
    {
        Half widthScale;
        Half heightScale;

        if (!GetInputScalar(operation, 1, OperandType::FLOAT16, widthScale, model, data) ||
            !GetInputScalar(operation, 2, OperandType::FLOAT16, heightScale, model, data))
        {
            return Fail("%s: Operation has invalid inputs for resizing by scale", __func__);
        }

        const TensorShape& inputShape = inputInfo.GetShape();
        armnnUtils::DataLayoutIndexed dataLayoutIndexed(descriptor.m_DataLayout);

        Half width  = static_cast<Half>(inputShape[dataLayoutIndexed.GetWidthIndex()]);
        Half height = static_cast<Half>(inputShape[dataLayoutIndexed.GetHeightIndex()]);

        descriptor.m_TargetWidth  = std::floor(width  * widthScale);
        descriptor.m_TargetHeight = std::floor(height * heightScale);
    }
    else
    {
        return Fail("%s: Operand has invalid data type for resizing by scale", __func__);
    }

    descriptor.m_AlignCorners     = GetOptionalBool(operation, 4, model, data);
    descriptor.m_HalfPixelCenters = GetOptionalBool(operation, 5, model, data);

    bool isSupported = false;
    armnn::BackendId setBackend;
    auto validateFunc = [&](const armnn::TensorInfo& outputInfo, bool& isSupported)
    {
        FORWARD_LAYER_SUPPORT_FUNC(__func__,
                                   IsResizeSupported,
                                   data.m_Backends,
                                   isSupported,
                                   setBackend,
                                   inputInfo,
                                   outputInfo,
                                   descriptor);
        };

    if(IsDynamicTensor(outputInfo))
    {
        isSupported = AreDynamicTensorsSupported();
    }
    else
    {
        validateFunc(outputInfo, isSupported);
    }

    if (!isSupported)
    {
        return false;
    }

    IConnectableLayer* layer = data.m_Network->AddResizeLayer(descriptor);
    layer->SetBackendId(setBackend);
    assert(layer != nullptr);
    input.Connect(layer->GetInputSlot(0));

    return SetupAndTrackLayerOutputSlot(operation, 0, *layer, model, data, nullptr, validateFunc);
}

bool Converter::ConvertSpaceToBatchNd(const Operation& operation, const Model& model, ConversionData& data)
{
    VLOG(DRIVER) << "Converter::ConvertSpaceToBatchNd()";

    LayerInputHandle input = ConvertToLayerInputHandle(operation, 0, model, data);
    if(!input.IsValid())
    {
        return Fail("%s: Operation has invalid inputs", __func__);
    }

    const armnn::TensorInfo &inputInfo = input.GetTensorInfo();
    unsigned int rank = inputInfo.GetNumDimensions();
    unsigned int spatialDim = rank - 2;

    if(rank != 4)
    {
        Fail("%s: Only inputs with rank 4 are supported", __func__);
    }

    const Operand *output = GetOutputOperand(operation, 0, model);
    if(!output)
    {
        return Fail("%s: Could not read output 0", __func__);
    }

    const armnn::TensorInfo &outputInfo = GetTensorInfoForOperand(*output);

    const Operand *blockShapeOperand = GetInputOperand(operation, 1, model);
    const Operand *paddingsOperand = GetInputOperand(operation, 2, model);

    armnn::TensorShape blockShapeOperandShape = GetTensorShapeForOperand(*blockShapeOperand);
    if(blockShapeOperandShape.GetNumDimensions() != 1 || blockShapeOperandShape.GetNumElements() != spatialDim)
    {
        return Fail("%s: Operation has invalid block shape operand: expected shape [%d]", __func__, spatialDim);
    }

    std::vector<int32_t> blockShape;
    if(!GetTensorInt32Values(*blockShapeOperand, blockShape, model, data))
    {
        return Fail("%s: Operation has an invalid or unsupported block size operand", __func__);
    }
    if(std::any_of(blockShape.cbegin(), blockShape.cend(), [](int32_t i)
    { return i < 1; }))
    {
        return Fail("%s: Block shape must be at least 1 in all dimensions.", __func__);
    }

    armnn::TensorShape paddingsOperandShape = GetTensorShapeForOperand(*paddingsOperand);
    if(paddingsOperandShape.GetNumDimensions() != 2 || paddingsOperandShape.GetNumElements() != 2 * spatialDim)
    {
        return Fail("%s: Operation has invalid paddings operand: expected shape [%d, 2]", __func__, spatialDim);
    }

    std::vector<std::pair<unsigned int, unsigned int>> paddingList;
    std::vector<int32_t> paddings;
    if(!GetTensorInt32Values(*paddingsOperand, paddings, model, data))
    {
        return Fail("%s: Operation has an invalid or unsupported paddings operand", __func__);
    }
    for (unsigned int i = 0; i < paddings.size() - 1; i += 2)
    {
        int paddingBeforeInput = paddings[i];
        int paddingAfterInput = paddings[i + 1];
        if(paddingBeforeInput < 0 || paddingAfterInput < 0)
        {
            return Fail("%s: Operation has invalid paddings operand, invalid padding values.", __func__);
        }

        paddingList.emplace_back((unsigned int) paddingBeforeInput, (unsigned int) paddingAfterInput);
    }

    armnn::SpaceToBatchNdDescriptor descriptor;
    descriptor.m_DataLayout = armnn::DataLayout::NHWC;
    descriptor.m_BlockShape.assign(blockShape.cbegin(), blockShape.cend());
    descriptor.m_PadList.assign(paddingList.cbegin(), paddingList.cend());

    if(Is12OrLaterOperand(*output))
    {
        descriptor.m_DataLayout = OptionalDataLayout(operation, 3, model, data);
    }

    bool isSupported = false;
    armnn::BackendId setBackend;
    auto validateFunc = [&](const armnn::TensorInfo &outputInfo, bool &isSupported)
    {
        FORWARD_LAYER_SUPPORT_FUNC(__func__,
                                   IsSpaceToBatchNdSupported,
                                   data.m_Backends,
                                   isSupported,
                                   setBackend,
                                   inputInfo,
                                   outputInfo,
                                   descriptor);
    };

    if(IsDynamicTensor(outputInfo))
    {
        isSupported = AreDynamicTensorsSupported();
    } else
    {
        validateFunc(outputInfo, isSupported);
    }

    if(!isSupported)
    {
        return false;
    }

    armnn::IConnectableLayer *const layer = data.m_Network->AddSpaceToBatchNdLayer(descriptor);
    layer->SetBackendId(setBackend);
    assert(layer != nullptr);
    input.Connect(layer->GetInputSlot(0));

    return SetupAndTrackLayerOutputSlot(operation, 0, *layer, model, data, nullptr, validateFunc);
}

bool Converter::ConvertSpaceToDepth(const Operation& operation, const Model& model, ConversionData& data)
{
    VLOG(DRIVER) << "Converter::ConvertSpaceToDepth()";

    LayerInputHandle input = ConvertToLayerInputHandle(operation, 0, model, data);
    if (!input.IsValid() )
    {
        return Fail("%s: Operation has invalid inputs", __func__);
    }

    const TensorInfo& inputInfo = input.GetTensorInfo();
    unsigned int rank = inputInfo.GetNumDimensions();
    if (rank != 4)
    {
        return Fail("%s: Only inputs with rank 4 are supported", __func__);
    }

    const Operand* output = GetOutputOperand(operation, 0, model);
    if (!output)
    {
        return Fail("%s: Could not read output 0", __func__);
    }

    const TensorInfo& outputInfo = GetTensorInfoForOperand(*output);

    SpaceToDepthDescriptor desc;

    GetInputScalar(operation, 1, OperandType::INT32, desc.m_BlockSize, model, data);

    if (desc.m_BlockSize <= 1)
    {
        return Fail("%s: Block size must be at least 1 in all dimensions");
    }

    desc.m_DataLayout = OptionalDataLayout(operation, 2, model, data);

    bool isSupported = false;
    armnn::BackendId setBackend;
    auto validateFunc = [&](const armnn::TensorInfo& outputInfo, bool& isSupported)
    {
        FORWARD_LAYER_SUPPORT_FUNC(__func__,
                                   IsSpaceToDepthSupported,
                                   data.m_Backends,
                                   isSupported,
                                   setBackend,
                                   inputInfo,
                                   outputInfo,
                                   desc);
    };

    if(IsDynamicTensor(outputInfo))
    {
        isSupported = AreDynamicTensorsSupported();
    }
    else
    {
        validateFunc(outputInfo, isSupported);
    }

    if (!isSupported)
    {
        return false;
    }

    IConnectableLayer* const layer = data.m_Network->AddSpaceToDepthLayer(desc);
    layer->SetBackendId(setBackend);
    assert(layer != nullptr);
    input.Connect(layer->GetInputSlot(0));

    return SetupAndTrackLayerOutputSlot(operation, 0, *layer, model, data, nullptr, validateFunc);
}

bool Converter::ConvertSoftmax(const Operation& operation, const Model& model, ConversionData& data)
{
    VLOG(DRIVER) << "Converter::ConvertSoftmax()";

    LayerInputHandle input = ConvertToLayerInputHandle(operation, 0, model, data);
    if (!input.IsValid())
    {
        return Fail("%s: Operation has invalid inputs", __func__);
    }

    const Operand* outputOperand = GetOutputOperand(operation, 0, model);
    if (!outputOperand)
    {
        return Fail("%s: Operation has no outputs", __func__);
    }

    const TensorInfo& outputInfo = GetTensorInfoForOperand(*outputOperand);

    SoftmaxDescriptor desc;
    OperandType outputType = outputOperand->type;

    // Read beta value
    if (outputType == OperandType::TENSOR_FLOAT16)
    {
        Half value;

        if (!GetInputScalar(operation, 1, OperandType::FLOAT16, value, model, data))
        {
            return Fail("%s: Operation has invalid inputs %d", __func__, outputType);
        }

        desc.m_Beta = static_cast<float>(value);
    }
    else
    {
        if (!GetInputFloat32(operation, 1, desc.m_Beta, model, data))
        {
            return Fail("%s: Operation has invalid inputs %d", __func__, outputType);
        }
    }

    if (operation.inputs.size() > 2 && !GetInputScalar(operation,
                                                                  2,
                                                                  OperandType::INT32,
                                                                  desc.m_Axis,
                                                                  model,
                                                                  data))
    {
        return Fail("%s: Operation has invalid inputs", __func__);
    }

    bool isSupported = false;
    armnn::BackendId setBackend;
    auto validateFunc = [&](const armnn::TensorInfo& outputInfo, bool& isSupported)
    {
        FORWARD_LAYER_SUPPORT_FUNC(__func__,
                                   IsSoftmaxSupported,
                                   data.m_Backends,
                                   isSupported,
                                   setBackend,
                                   input.GetTensorInfo(),
                                   outputInfo,
                                   desc);
        };

    if(IsDynamicTensor(outputInfo))
    {
        isSupported = AreDynamicTensorsSupported();
    }
    else
    {
        validateFunc(outputInfo, isSupported);
    }

    if (!isSupported)
    {
        return false;
    }

    IConnectableLayer* layer = data.m_Network->AddSoftmaxLayer(desc);
    layer->SetBackendId(setBackend);
    assert(layer != nullptr);
    input.Connect(layer->GetInputSlot(0));

    return SetupAndTrackLayerOutputSlot(operation, 0, *layer, model, data, nullptr, validateFunc);
}

bool Converter::ConvertSub(const Operation& operation, const Model& model, ConversionData& data)
{
    VLOG(DRIVER) << "Converter::ConvertSub()";

    LayerInputHandle input0 = ConvertToLayerInputHandle(operation, 0, model, data);
    LayerInputHandle input1 = ConvertToLayerInputHandle(operation, 1, model, data);

    if (!input0.IsValid() || !input1.IsValid())
    {
        return Fail("%s: Operation has invalid inputs", __func__);
    }

    // The FuseActivation parameter is always the input index 2
    // and it should be optional
    ActivationFn activationFunction;
    if (!GetOptionalInputActivation(operation, 2, activationFunction, model, data))
    {
        return Fail("%s: Operation has invalid inputs", __func__);
    }

    const Operand* output = GetOutputOperand(operation, 0, model);
    if (!output)
    {
        return Fail("%s: Could not read output 0", __func__);
    }

    const armnn::TensorInfo& outputInfo = GetTensorInfoForOperand(*output);

    bool isSupported = false;
    armnn::BackendId setBackend;
    auto validateFunc = [&](const armnn::TensorInfo& outputInfo, bool& isSupported)
    {
        ARMNN_NO_DEPRECATE_WARN_BEGIN
        FORWARD_LAYER_SUPPORT_FUNC(__func__,
                                   IsSubtractionSupported,
                                   data.m_Backends,
                                   isSupported,
                                   setBackend,
                                   input0.GetTensorInfo(),
                                   input1.GetTensorInfo(),
                                   outputInfo);
        ARMNN_NO_DEPRECATE_WARN_END
    };

    if(IsDynamicTensor(outputInfo))
    {
        isSupported = AreDynamicTensorsSupported();
    }
    else
    {
        validateFunc(outputInfo, isSupported);
    }

    if (!isSupported)
    {
        return false;
    }

    ARMNN_NO_DEPRECATE_WARN_BEGIN
    armnn::IConnectableLayer* const startLayer = data.m_Network->AddSubtractionLayer();
    ARMNN_NO_DEPRECATE_WARN_END
    startLayer->SetBackendId(setBackend);

    bool isReshapeSupported = BroadcastTensor(input0, input1, startLayer, data);
    if (!isReshapeSupported)
    {
        return false;
    }
    return SetupAndTrackLayerOutputSlot(operation, 0, *startLayer, model,
                                        data, nullptr, validateFunc, activationFunction);
}

bool Converter::ConvertTanH(const Operation& operation, const Model& model, ConversionData& data)
{
    VLOG(DRIVER) << "Converter::ConvertTanH()";

    armnn::ActivationDescriptor desc;
    desc.m_Function = armnn::ActivationFunction::TanH;
    desc.m_A = 1.0f; // android nn does not support tanH parameters
    desc.m_B = 1.0f; // set to 1.0f for unity scaling

    return ConvertToActivation(operation, __func__, desc, model, data);
}

bool Converter::ConvertTransposeConv2d(const Operation& operation, const Model& model, ConversionData& data)
{
    VLOG(DRIVER) << "Converter::ConvertTransposeConv2d()";

    LayerInputHandle input = ConvertToLayerInputHandle(operation, 0, model, data);

    if (!input.IsValid())
    {
        return Fail("%s: Operation has invalid inputs", __func__);
    }

    const Operand* output = GetOutputOperand(operation, 0, model);

    if (!output)
    {
        return Fail("%s: Could not read output 0", __func__);
    }

    const TensorInfo& inputInfo  = input.GetTensorInfo();
    const TensorInfo& outputInfo = GetTensorInfoForOperand(*output);

    // ArmNN does not currently support non-fixed weights or bias
    // Find the shape of the weights tensor. In AndroidNN this will be [ 1, H, W, I * M ]
    const Operand* weightsOperand = GetInputOperand(operation, 1, model);

    if (weightsOperand == nullptr)
    {
        return Fail("%s: Operand is invalid", __func__);
    }
    TransposeConvolution2dDescriptor desc;
    desc.m_DataLayout = DataLayout::NHWC;

    // Determine whether padding is implicit or explicit
    bool implicitPadding = operation.inputs.size() == 9;

    if (implicitPadding )
    {
        desc.m_DataLayout = OptionalDataLayout(operation, 8, model, data);
    }
    else
    {
        desc.m_DataLayout = OptionalDataLayout(operation, 10, model, data);
    }

    armnnUtils::DataLayoutIndexed dataLayoutIndexed(desc.m_DataLayout);
    unsigned int widthIndex = dataLayoutIndexed.GetWidthIndex();
    unsigned int heightIndex = dataLayoutIndexed.GetHeightIndex();

    const PermutationVector OHWIToOIHW = {0, 2, 3, 1};

    // The shape of the weight is [depth_out, filter_height, filter_width, depth_in].
    // We have to permute it to OIHW if the data layout is NCHW.
    const ConstTensorPin weightsPin = (desc.m_DataLayout == DataLayout::NCHW) ?
                                      ConvertOperationInputToConstTensorPin(operation, 1,
                                                                                       model, data, OHWIToOIHW) :
                                      ConvertOperationInputToConstTensorPin(operation, 1, model, data);

    // Bias is a 1D tensor
    const ConstTensorPin biasPin =
        ConvertOperationInputToConstTensorPin(operation, 2, model, data);

    if (!weightsPin.IsValid())
    {
        return Fail("%s: Operation has invalid weights", __func__);
    }

    if (!biasPin.IsValid())
    {
        return Fail("%s: Operation has invalid biases", __func__);
    }

    ConstTensor weights = weightsPin.GetConstTensor();
    ConstTensor bias = biasPin.GetConstTensor();
    SanitizeBiasQuantizationScale(bias.GetInfo(), weights.GetInfo(), inputInfo);

    ActivationFn activation;

    if (implicitPadding)
    {
        int32_t strideX{0};
        int32_t strideY{0};
        int32_t padLeft{0};
        int32_t padRight{0};
        int32_t padTop{0};
        int32_t padBottom{0};

        ::android::nn::PaddingScheme paddingScheme;
        if (!GetInputPaddingScheme(operation, 4, paddingScheme, model, data) ||
            !GetInputScalar(operation, 5, OperandType::INT32, strideX, model, data) ||
            !GetInputScalar(operation, 6, OperandType::INT32, strideY, model, data) ||
            !GetInputActivationFunction(operation, 7, activation, model, data))
        {
            return Fail("%s: Operation has invalid inputs (implicit padding)", __func__);
        }

        const uint32_t kernelX = weights.GetShape()[widthIndex];
        const uint32_t kernelY = weights.GetShape()[heightIndex];

        // If output shape has been specified as a parameter then extract it and make it available.
        const Operand* outputShapeOperand = GetInputOperand(operation, 3, model, false);
        std::vector<int32_t> outputShape;
        if ((outputShapeOperand) && (GetTensorInt32Values(*outputShapeOperand, outputShape, model, data)))
        {
            // Change from signed to unsigned int to store in TransposeConvolution2dDescriptor.
            for (int dimension : outputShape)
            {
                desc.m_OutputShape.push_back(static_cast<unsigned int>(dimension));
            }
            desc.m_OutputShapeEnabled = true;
        }

        uint32_t outputX;
        uint32_t outputY;

        if (IsDynamicTensor(outputInfo))
        {
            if (outputShape.size() == 0)
            {
                return Fail("%s: Padding sizes cannot be inferred", __func__);
            }

            outputX = outputShape[widthIndex];
            outputY = outputShape[heightIndex];
        }
        else
        {
            outputX = outputInfo.GetShape()[widthIndex];
            outputY = outputInfo.GetShape()[heightIndex];
        }

        CalcPaddingTransposeConv(outputX, kernelX, strideX, padLeft, padRight, paddingScheme);
        CalcPaddingTransposeConv(outputY, kernelY, strideY, padTop, padBottom, paddingScheme);

        // NOTE: The Android NN API allows for negative padding values in TransposeConv2d,
        // but Arm NN only supports values >= 0
        if (padLeft < 0 || padRight < 0 || padTop < 0 || padBottom < 0)
        {
            return Fail("%s: Negative padding values are not supported", __func__);
        }

        desc.m_StrideX   = armnn::numeric_cast<uint32_t>(strideX);
        desc.m_StrideY   = armnn::numeric_cast<uint32_t>(strideY);
        desc.m_PadLeft   = armnn::numeric_cast<uint32_t>(padLeft);
        desc.m_PadRight  = armnn::numeric_cast<uint32_t>(padRight);
        desc.m_PadTop    = armnn::numeric_cast<uint32_t>(padTop);
        desc.m_PadBottom = armnn::numeric_cast<uint32_t>(padBottom);
    }
    else if (operation.inputs.size() == 11)
    {
        // explicit padding
        if (!GetInputScalar(operation, 3, OperandType::INT32, desc.m_PadLeft, model, data) ||
            !GetInputScalar(operation, 4, OperandType::INT32, desc.m_PadRight, model, data) ||
            !GetInputScalar(operation, 5, OperandType::INT32, desc.m_PadTop, model, data) ||
            !GetInputScalar(operation, 6, OperandType::INT32, desc.m_PadBottom, model, data) ||
            !GetInputScalar(operation, 7, OperandType::INT32, desc.m_StrideX, model, data) ||
            !GetInputScalar(operation, 8, OperandType::INT32, desc.m_StrideY, model, data) ||
            !GetInputActivationFunction(operation,  9, activation, model, data))
        {
            return Fail("%s: Operation has invalid inputs (explicit padding)", __func__);
        }
    }
    else
    {
        return Fail("%s: Unsupported number of operation inputs", __func__);
    }

    desc.m_BiasEnabled = true;
    Optional<TensorInfo> biases(bias.GetInfo());

    bool isSupported = false;
    armnn::BackendId setBackend;
    auto validateFunc = [&](const armnn::TensorInfo& outputInfo, bool& isSupported)
    {
        FORWARD_LAYER_SUPPORT_FUNC(__func__,
                                   IsTransposeConvolution2dSupported,
                                   data.m_Backends,
                                   isSupported,
                                   setBackend,
                                   inputInfo,
                                   outputInfo,
                                   desc,
                                   weights.GetInfo(),
                                   biases);
    };

    if(IsDynamicTensor(outputInfo))
    {
        isSupported = AreDynamicTensorsSupported();
    }
    else
    {
        validateFunc(outputInfo, isSupported);
    }
    if (!isSupported)
    {
        return false;
    }

    IConnectableLayer* startLayer =
        data.m_Network->AddTransposeConvolution2dLayer(desc, weights, Optional<ConstTensor>(bias));
    startLayer->SetBackendId(setBackend);
    if (!startLayer)
    {
        return Fail("%s: AddTransposeConvolution2dLayer failed", __func__);
    }

    input.Connect(startLayer->GetInputSlot(0));

    return SetupAndTrackLayerOutputSlot(operation, 0, *startLayer, model,
                                                   data, nullptr, validateFunc, activation);
}

bool Converter::ConvertSqrt(const Operation& operation, const Model& model, ConversionData& data)
{
    VLOG(DRIVER) << "Converter::ConvertSqrt()";
    ActivationDescriptor desc;
    desc.m_Function = ActivationFunction::Sqrt;

    return ::ConvertToActivation(operation, __func__, desc, model, data);
}

bool Converter::ConvertSqueeze(const Operation& operation, const Model& model, ConversionData& data)
{
    VLOG(DRIVER) << "Converter::ConvertSqueeze()";

    LayerInputHandle input = ConvertToLayerInputHandle(operation, 0, model, data);
    if (!input.IsValid())
    {
        return Fail("%s: Operation has invalid inputs", __func__);
    }

    const armnn::TensorInfo& inputInfo  = input.GetTensorInfo();
    unsigned int rank = inputInfo.GetNumDimensions();
    if (rank > 4)
    {
        Fail("%s: Inputs with rank greater than 4 are not supported", __func__);
    }

    const Operand* output = GetOutputOperand(operation, 0, model);
    if (!output)
    {
        return Fail("%s: Could not read output 0", __func__);
    }

    if (IsDynamicTensor(GetTensorInfoForOperand(*output)) && !(AreDynamicTensorsSupported()))
    {
        return Fail("%s: Dynamic output tensors are not supported", __func__);
    }

    // NOTE: Axis is an optional parameter to SQUEEZE, therefore we do not want to generate a failure
    // if the operand index is out of bounds.
    const Operand* axisOperand = GetInputOperand(operation, 1, model, false);

    const uint32_t dimensionSequence[] = { 0, 1, 2, 3 };

    std::vector<int32_t> axis;
    if (!axisOperand)
    {
        axis.assign(dimensionSequence,
                    dimensionSequence + rank);
    }
    else if (!GetTensorInt32Values(*axisOperand, axis, model, data))
    {
        return Fail("%s: Operation has an invalid or unsupported axis operand", __func__);
    }

    std::vector<uint32_t> outputDims;
    for (unsigned int i = 0; i < rank; i++)
    {
        bool skipSqueeze = (std::find(axis.begin(), axis.end(), i) == axis.end());
        auto currentDimension = inputInfo.GetShape()[i];
        if (skipSqueeze || currentDimension != 1)
        {
            outputDims.push_back(currentDimension);
        }
    }

    armnn::TensorShape outShape = armnn::TensorShape(outputDims.size(), outputDims.data());

    armnn::TensorInfo outputInfo = inputInfo;
    outputInfo.SetShape(outShape);

    armnn::ReshapeDescriptor reshapeDesc;
    reshapeDesc.m_TargetShape = outputInfo.GetShape();

    bool isSupported = false;
    armnn::BackendId setBackend;
    FORWARD_LAYER_SUPPORT_FUNC(__func__,
                               IsReshapeSupported,
                               data.m_Backends,
                               isSupported,
                               setBackend,
                               inputInfo,
                               outputInfo,
                               reshapeDesc);

    if (!isSupported)
    {
        return false;
    }

    armnn::IConnectableLayer* const layer = data.m_Network->AddReshapeLayer(reshapeDesc);
    layer->SetBackendId(setBackend);
    assert(layer != nullptr);
    input.Connect(layer->GetInputSlot(0));

    return SetupAndTrackLayerOutputSlot(operation, 0, *layer, model, data);
}

bool Converter::ConvertStridedSlice(const Operation& operation, const Model& model, ConversionData& data)
{
    VLOG(DRIVER) << "Converter::ConvertStridedSlice()";

    LayerInputHandle input = ConvertToLayerInputHandle(operation, 0, model, data);
    if (!input.IsValid())
    {
        return Fail("%s: Operation has invalid inputs", __func__);
    }

    const armnn::TensorInfo& inputInfo = input.GetTensorInfo();
    unsigned int rank = inputInfo.GetNumDimensions();
    if (rank > 4)
    {
        Fail("%s: Inputs with rank greater than 4 are not supported", __func__);
    }

    const Operand* output = GetOutputOperand(operation, 0, model);
    if (!output)
    {
        return Fail("%s: Could not read output 0", __func__);
    }

    const armnn::TensorInfo& outputInfo = GetTensorInfoForOperand(*output);

    const Operand* beginOperand   = GetInputOperand(operation, 1, model);
    const Operand* endOperand     = GetInputOperand(operation, 2, model);
    const Operand* stridesOperand = GetInputOperand(operation, 3, model);

    std::vector<int32_t> beginValues;
    std::vector<int32_t> endValues;
    std::vector<int32_t> stridesValues;

    // The length of the beginOperand, endOperand and stridesOperand must be of a rank(input)
    auto ValidateInputOperands = [&] (const Operand& operand, std::vector<int32_t>& operandValues)
    {
        if (!GetTensorInt32Values(operand, operandValues, model, data))
        {
            return false;
        }

        if (operandValues.size() != rank)
        {
            return false;
        }

        return true;
    };

    if (!ValidateInputOperands(*beginOperand, beginValues)
        || !ValidateInputOperands(*endOperand, endValues)
        || !ValidateInputOperands(*stridesOperand, stridesValues))
    {
        return Fail("%s: Operation has invalid input operand", __func__);
    }

    // Stride cannot have value '0'
    if (std::any_of(stridesValues.cbegin(), stridesValues.cend(), [](int32_t i){ return i == 0; }))
    {
        return Fail("%s: Stride must be non-zero value.", __func__);
    }

    armnn::StridedSliceDescriptor descriptor;
    descriptor.m_Begin.assign(beginValues.cbegin(), beginValues.cend());
    descriptor.m_End.assign(endValues.cbegin(), endValues.cend());
    descriptor.m_Stride.assign(stridesValues.cbegin(), stridesValues.cend());
    descriptor.m_DataLayout = armnn::DataLayout::NHWC;

    // Get the "begin_mask", "end_mask", and "shrink_axis_mask" flags
    if (!GetInputInt32(operation, 4, descriptor.m_BeginMask, model, data) ||
        !GetInputInt32(operation, 5, descriptor.m_EndMask, model, data) ||
        !GetInputInt32(operation, 6, descriptor.m_ShrinkAxisMask, model, data))
    {
        return Fail("%s: Operation has invalid inputs", __func__);
    }

    bool isSupported = false;
    armnn::BackendId setBackend;
    auto validateFunc = [&](const armnn::TensorInfo& outputInfo, bool& isSupported)
    {
        FORWARD_LAYER_SUPPORT_FUNC(__func__,
                                   IsStridedSliceSupported,
                                   data.m_Backends,
                                   isSupported,
                                   setBackend,
                                   inputInfo,
                                   outputInfo,
                                   descriptor);
    };

    if(IsDynamicTensor(outputInfo))
    {
        isSupported = AreDynamicTensorsSupported();
    }
    else
    {
        validateFunc(outputInfo, isSupported);
    }

    if (!isSupported)
    {
        return false;
    }

    // Check if slice can fit in a inferred output
    armnn::TensorShape inputShape = inputInfo.GetShape();
    for (unsigned int i = 0; i < inputShape.GetNumDimensions(); i++)
    {
        int stride = descriptor.m_Stride[i];

        if (descriptor.m_ShrinkAxisMask & (1 << i))
        {
            // If the difference between the start point and the end point of the slice on an axis being shrunk
            // is greater than 1 then throw an error as the output will not be large enough to hold the slice
            if (((descriptor.m_Begin[i] - descriptor.m_End[i]) > 1)
                || ((descriptor.m_Begin[i] - descriptor.m_End[i]) < -1))
            {
                return Fail("%s: StridedSlice: Output will not be large enough to hold the slice", __func__);
            }

            if(stride < 0)
            {
                return Fail("%s: StridedSlice: Stride can not be negative while ShrinkAxisMask is set.", __func__);
            }
        }
    }

    armnn::IConnectableLayer* const layer = data.m_Network->AddStridedSliceLayer(descriptor);
    layer->SetBackendId(setBackend);
    assert(layer != nullptr);
    input.Connect(layer->GetInputSlot(0));

    return SetupAndTrackLayerOutputSlot(operation, 0, *layer, model, data, nullptr, validateFunc);
}

bool Converter::ConvertTranspose(const Operation& operation, const Model& model, ConversionData& data)
{
    VLOG(DRIVER) << "Converter::ConvertTranspose()";

    LayerInputHandle input = ConvertToLayerInputHandle(operation, 0, model, data);
    if (!input.IsValid())
    {
        return Fail("%s: Operation has invalid inputs", __func__);
    }

    const armnn::TensorInfo& inputInfo = input.GetTensorInfo();
    unsigned int rank = inputInfo.GetNumDimensions();
    if (rank > 4)
    {
        Fail("%s: Inputs with rank greater than 4 are not supported", __func__);
    }

    // NOTE: Axis is an optional parameter to TRANSPOSE, therefore we do not want to generate a failure
    // if the operand index is out of bounds.
    const Operand* permOperand = GetInputOperand(operation, 1, model, false);

    std::vector<int32_t> perm(rank);
    if (!permOperand || (permOperand->lifetime == OperandLifeTime::NO_VALUE))
    {
        for (unsigned int i = rank; i > 0; i--)
        {
            perm[rank - i] = armnn::numeric_cast<int> (i - 1);
        }
    }
    else if (!GetTensorInt32Values(*permOperand, perm, model, data))
    {
        return Fail("%s: Operation has an invalid or unsupported permutation operand", __func__);
    }

    std::vector<uint32_t> outputDims(perm.begin(), perm.begin() + rank);

    armnn::TransposeDescriptor transposeDesc;
    transposeDesc.m_DimMappings = armnn::PermutationVector(outputDims.data(), outputDims.size());

    const Operand* output = GetOutputOperand(operation, 0, model);
    if (!output)
    {
        return Fail("%s: Could not read output 0", __func__);
    }

    const armnn::TensorInfo& outputInfo = GetTensorInfoForOperand(*output);

    bool isSupported = false;
    armnn::BackendId setBackend;
    auto validateFunc = [&](const armnn::TensorInfo& outputInfo, bool& isSupported)
    {
        FORWARD_LAYER_SUPPORT_FUNC(__func__,
                                   IsTransposeSupported,
                                   data.m_Backends,
                                   isSupported,
                                   setBackend,
                                   inputInfo,
                                   outputInfo,
                                   transposeDesc);
    };

    if(IsDynamicTensor(outputInfo))
    {
        isSupported = AreDynamicTensorsSupported();
    }
    else
    {
        validateFunc(outputInfo, isSupported);
    }

    if (!isSupported)
    {
        return false;
    }

    armnn::IConnectableLayer* const layer = data.m_Network->AddTransposeLayer(transposeDesc);
    layer->SetBackendId(setBackend);
    assert(layer != nullptr);
    input.Connect(layer->GetInputSlot(0));

    return SetupAndTrackLayerOutputSlot(operation, 0, *layer, model, data, nullptr, validateFunc);
}

} // namespace armnn_driver
