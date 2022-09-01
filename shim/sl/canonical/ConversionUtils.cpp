//
// Copyright © 2022 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "ConversionUtils.hpp"
#include <armnnUtils/Permute.hpp>

///
/// Helper classes
///

namespace armnn_driver
{

LayerInputHandle::LayerInputHandle()
    : m_OutputSlot(nullptr)
    , m_Valid(false)
{}

LayerInputHandle::LayerInputHandle(bool valid, armnn::IOutputSlot* outputSlot, armnn::TensorInfo tensorInfo)
    : m_OutputSlot(outputSlot)
    , m_Valid(valid)
    , m_TensorInfo(tensorInfo)
{}

bool LayerInputHandle::IsValid() const
{
    return m_Valid;
}

void LayerInputHandle::Connect(armnn::IInputSlot& inputSlot)
{
    ARMNN_ASSERT(IsValid());
    if (m_OutputSlot)
    {
        m_OutputSlot->Connect(inputSlot);
    }
}

void LayerInputHandle::Disconnect(armnn::IInputSlot& inputSlot)
{
    ARMNN_ASSERT(IsValid());
    if (m_OutputSlot)
    {
        m_OutputSlot->Disconnect(inputSlot);
    }
}

const armnn::TensorInfo& LayerInputHandle::GetTensorInfo() const
{
    return m_TensorInfo;
}

void LayerInputHandle::SanitizeQuantizationScale(LayerInputHandle& weight, LayerInputHandle& input)
{
    if (m_OutputSlot)
    {
        armnn::TensorInfo weightInfo = weight.GetTensorInfo();
        armnn::TensorInfo inputInfo = input.GetTensorInfo();
        armnn::TensorInfo biasInfo = GetTensorInfo();

        SanitizeBiasQuantizationScale(biasInfo, weightInfo, inputInfo);

        m_TensorInfo = biasInfo;
        m_OutputSlot->SetTensorInfo(biasInfo);
    }
}

armnn::IOutputSlot* LayerInputHandle::GetOutputSlot() const
{
    return m_OutputSlot;
}

ConstTensorPin::ConstTensorPin(bool optional)
    : m_Optional(optional)
{}

ConstTensorPin::ConstTensorPin(armnn::TensorInfo& tensorInfo,
                               const void* valueStart,
                               uint32_t numBytes,
                               const armnn::PermutationVector& mappings)
    : m_Optional(false)
{
    armnn::IgnoreUnused(numBytes);
    if (tensorInfo.GetNumBytes() != numBytes)
    {
        VLOG(DRIVER) << "The size of ConstTensor does not match its TensorInfo.";
    }

    const bool needsSwizzling = (mappings.GetSize() > 0);
    if (needsSwizzling)
    {
        m_SwizzledTensorData.resize(tensorInfo.GetNumBytes());
        SwizzleAndroidNn4dTensorToArmNn(tensorInfo, valueStart, m_SwizzledTensorData.data(), mappings);

        m_ConstTensor = armnn::ConstTensor(tensorInfo, m_SwizzledTensorData.data());
    }
    else
    {
        m_ConstTensor = armnn::ConstTensor(tensorInfo, valueStart);
    }
}

bool ConstTensorPin::IsValid() const
{
    return m_ConstTensor.GetMemoryArea() != nullptr;
}

bool ConstTensorPin::IsOptional() const
{
    return m_Optional;
}

const armnn::ConstTensor& ConstTensorPin::GetConstTensor() const
{
    return m_ConstTensor;
}

const armnn::ConstTensor* ConstTensorPin::GetConstTensorPtr() const
{
    if (IsValid() && m_ConstTensor.GetNumElements() > 0)
    {
        return &m_ConstTensor;
    }
    // tensor is either invalid, or has no elements (indicating an optional tensor that was not provided)
    return nullptr;
}

///
/// Utility functions
///

bool IsWeightsValid(const Operation& operation,
                    uint32_t inputIndex,
                    const Model& model)
{
    const Operand* operand = GetInputOperand(operation, inputIndex, model);
    if (!operand)
    {
        Fail("%s: failed to get input operand %i", __func__, inputIndex);
        return false;
    }

    if (operand->lifetime    != OperandLifeTime::CONSTANT_COPY
        && operand->lifetime != OperandLifeTime::CONSTANT_REFERENCE
        && operand->lifetime != OperandLifeTime::NO_VALUE)
    {
        return false;
    }
    return true;
}

ConstTensorPin ConvertOperandToConstTensorPin(const Operand& operand,
                                              const Model& model,
                                              const ConversionData& data,
                                              const armnn::PermutationVector& dimensionMappings,
                                              const armnn::TensorShape* overrideTensorShape,
                                              bool optional,
                                              const armnn::DataType* overrideDataType)
{
    if (!IsOperandTypeSupportedForTensors(operand.type))
    {
        VLOG(DRIVER) << __func__ << ": unsupported operand type for tensor" << operand.type;
        return ConstTensorPin();
    }

    if (!optional && !IsOperandConstant(operand))
    {
        VLOG(DRIVER) << __func__ << ": lifetime for input tensor: r" << operand.lifetime;
        return ConstTensorPin();
    }

    const void* const valueStart = GetOperandValueReadOnlyAddress(operand, model, data, optional);
    if (!valueStart)
    {
        if (optional)
        {
            // optional tensor with no values is not really an error; return it as invalid, but marked as optional
            return ConstTensorPin(true);
        }
        // mandatory tensor with no values
        Fail("%s: failed to get operand address", __func__);
        return ConstTensorPin();
    }

    armnn::TensorInfo tensorInfo = GetTensorInfoForOperand(operand);

    if (overrideTensorShape)
    {
        tensorInfo.SetShape(*overrideTensorShape);
    }

    if (overrideDataType)
    {
        tensorInfo.SetDataType(*overrideDataType);
    }

    // Make sure isConstant flag is set.
    tensorInfo.SetConstant();
    return ConstTensorPin(tensorInfo, valueStart, operand.location.length, dimensionMappings);
}

LayerInputHandle ConvertToLayerInputHandle(const Operation& operation,
                                           uint32_t inputIndex,
                                           const Model& model,
                                           ConversionData& data,
                                           const armnn::PermutationVector& dimensionMappings,
                                           const LayerInputHandle* inputHandle)
{

    const Operand* operand = GetInputOperand(operation, inputIndex, model);
    if (!operand)
    {
        Fail("%s: failed to get input operand %i", __func__, inputIndex);
        return LayerInputHandle();
    }

    if (!IsOperandTypeSupportedForTensors(operand->type))
    {
        VLOG(DRIVER) << __func__ << ": unsupported operand type for tensor: " << operand->type;
        return LayerInputHandle();
    }

    try
    {
        armnn::TensorInfo operandTensorInfo = GetTensorInfoForOperand(*operand);

        if (IsDynamicTensor(operandTensorInfo))
        {
            data.m_DynamicInputsEncountered = true;

            const uint32_t operandIndex = operation.inputs[inputIndex];

            // Check if the dynamic input tensors have been inferred by one of the previous layers
            // If not we can't support them
            if (data.m_OutputSlotForOperand.size() >= operandIndex && data.m_OutputSlotForOperand[operandIndex])
            {
                operandTensorInfo = data.m_OutputSlotForOperand[operandIndex]->GetTensorInfo();
            }
            else
            {
                Fail("%s: Type 2 dynamic input tensors are not supported", __func__);
                return LayerInputHandle();
            }
        }

        switch (operand->lifetime)
        {
            case OperandLifeTime::SUBGRAPH_INPUT:
            {
                // NOTE: We must check whether we can support the input tensor on at least one
                // of the provided backends; otherwise we cannot convert the operation
                bool isInputSupported = false;
                FORWARD_LAYER_SUPPORT_FUNC(__func__,
                                           IsInputSupported,
                                           data.m_Backends,
                                           isInputSupported,
                                           armnn::BackendId(),
                                           operandTensorInfo);

                if (!isInputSupported)
                {
                    Fail("%s: unsupported input tensor", __func__);
                    return LayerInputHandle();
                }

                [[clang::fallthrough]]; // intentional fallthrough
            }
            case OperandLifeTime::TEMPORARY_VARIABLE: // intentional fallthrough
            case OperandLifeTime::SUBGRAPH_OUTPUT:
            {
                // The tensor is either an operand internal to the model, or a model input.
                // It can be associated with an ArmNN output slot for an existing layer.

                // m_OutputSlotForOperand[...] can be nullptr if the previous layer could not be converted
                const uint32_t operandIndex = operation.inputs[inputIndex];
                return LayerInputHandle(true, data.m_OutputSlotForOperand[operandIndex], operandTensorInfo);
            }
            case OperandLifeTime::CONSTANT_COPY: // intentional fallthrough
            case OperandLifeTime::POINTER:
            case OperandLifeTime::CONSTANT_REFERENCE:
            {
                auto constantTensorDataType = operandTensorInfo.GetDataType();
                // The tensor has an already known constant value, and can be converted into an ArmNN Constant layer.
                ConstTensorPin tensorPin = ConvertOperandToConstTensorPin(*operand,
                                                                          model,
                                                                          data,
                                                                          dimensionMappings,
                                                                          nullptr,
                                                                          false,
                                                                          &constantTensorDataType);
                if (tensorPin.IsValid())
                {
                    bool isSupported = false;
                    armnn::BackendId setBackend;
                    FORWARD_LAYER_SUPPORT_FUNC(__func__,
                                               IsConstantSupported,
                                               data.m_Backends,
                                               isSupported,
                                               setBackend,
                                               tensorPin.GetConstTensor().GetInfo());
                    if (!isSupported)
                    {
                        return LayerInputHandle();
                    }

                    armnn::IConnectableLayer* constantLayer =
                        data.m_Network->AddConstantLayer(tensorPin.GetConstTensor());
                    constantLayer->SetBackendId(setBackend);
                    armnn::IOutputSlot& outputSlot = constantLayer->GetOutputSlot(0);
                    armnn::TensorInfo constantTensorInfo = tensorPin.GetConstTensor().GetInfo();
                    outputSlot.SetTensorInfo(constantTensorInfo);

                    return LayerInputHandle(true, &outputSlot, constantTensorInfo);
                }
                else
                {
                    Fail("%s: invalid operand tensor", __func__);
                    return LayerInputHandle();
                }
                break;
            }
            default:
            {
                VLOG(DRIVER) << __func__ << ": unsupported lifetime for input tensor: " << operand->lifetime;
                return LayerInputHandle();
            }
        }
    }
    catch (UnsupportedOperand<OperandType>& e)
    {
        VLOG(DRIVER) << __func__ << ": Operand type: " << e.m_type << " not supported in ArmnnDriver";
        return LayerInputHandle();
    }
}

bool ConvertPaddings(const Operation& operation,
                     const Model& model,
                     ConversionData& data,
                     unsigned int rank,
                     armnn::PadDescriptor& padDescriptor)
{
    const Operand* paddingsOperand = GetInputOperand(operation, 1, model);
    if (!paddingsOperand)
    {
        return Fail("%s: Could not read paddings operand", __func__);
    }

    armnn::TensorShape paddingsOperandShape = GetTensorShapeForOperand(*paddingsOperand);
    if (paddingsOperandShape.GetNumDimensions() != 2 || paddingsOperandShape.GetNumElements() != rank * 2)
    {
        return Fail("%s: Operation has invalid paddings operand: expected shape [%d, 2]",  __func__, rank);
    }

    std::vector<int32_t> paddings;
    if (!GetTensorInt32Values(*paddingsOperand, paddings, model, data))
    {
        return Fail("%s: Operation has invalid or unsupported paddings operand", __func__);
    }

    // add padding for each dimension of input tensor.
    for (unsigned int i = 0; i < paddings.size() - 1; i += 2)
    {
        int paddingBeforeInput = paddings[i];
        int paddingAfterInput  = paddings[i + 1];

        if (paddingBeforeInput < 0 || paddingAfterInput < 0)
        {
            return Fail("%s: Operation has invalid paddings operand, invalid padding values.",  __func__);
        }

        padDescriptor.m_PadList.emplace_back((unsigned int) paddingBeforeInput, (unsigned int) paddingAfterInput);
    }

    return true;
}


bool ConvertPooling2d(const Operation& operation,
                      const char* operationName,
                      armnn::PoolingAlgorithm poolType,
                      const Model& model,
                      ConversionData& data)
{

    VLOG(DRIVER) << "Converter::ConvertL2Pool2d()";

    LayerInputHandle input = ConvertToLayerInputHandle(operation, 0, model, data);
    if (!input.IsValid())
    {
        return Fail("%s: Operation Could not read input 0", operationName);
    }

    const Operand* output = GetOutputOperand(operation, 0, model);
    if (!output)
    {
        return Fail("%s: Could not read output 0", __func__);
    }

    const armnn::TensorInfo& inputInfo  = input.GetTensorInfo();
    const armnn::TensorInfo& outputInfo = GetTensorInfoForOperand(*output);

    armnn::Pooling2dDescriptor desc;
    desc.m_PoolType = poolType;
    desc.m_OutputShapeRounding = armnn::OutputShapeRounding::Floor;
    desc.m_DataLayout = armnn::DataLayout::NHWC;

    ActivationFn activation;

    auto inputSize = operation.inputs.size();

    if (inputSize >= 10)
    {
        // one input, 9 parameters (padding l r t b, stridex, stridey, width, height, activation type)
        if (!GetInputScalar(operation, 1, OperandType::INT32, desc.m_PadLeft, model, data) ||
            !GetInputScalar(operation, 2, OperandType::INT32, desc.m_PadRight, model, data) ||
            !GetInputScalar(operation, 3, OperandType::INT32, desc.m_PadTop, model, data) ||
            !GetInputScalar(operation, 4, OperandType::INT32, desc.m_PadBottom, model, data) ||
            !GetInputScalar(operation, 5, OperandType::INT32, desc.m_StrideX, model, data) ||
            !GetInputScalar(operation, 6, OperandType::INT32, desc.m_StrideY, model, data) ||
            !GetInputScalar(operation, 7, OperandType::INT32, desc.m_PoolWidth, model, data) ||
            !GetInputScalar(operation, 8, OperandType::INT32, desc.m_PoolHeight, model, data) ||
            !GetInputActivationFunction(operation, 9, activation, model, data))
        {
            return Fail("%s: Operation has invalid inputs", operationName);
        }

        if (Is12OrLaterOperand(*output))
        {
            desc.m_DataLayout = OptionalDataLayout(operation, 10, model, data);
        }
    }
    else
    {
        // one input, 6 parameters (padding, stridex, stridey, width, height, activation type)
        ::android::nn::PaddingScheme scheme;
        if (!GetInputPaddingScheme(operation, 1, scheme, model, data) ||
            !GetInputScalar(operation, 2, OperandType::INT32, desc.m_StrideX, model, data) ||
            !GetInputScalar(operation, 3, OperandType::INT32, desc.m_StrideY, model, data) ||
            !GetInputScalar(operation, 4, OperandType::INT32, desc.m_PoolWidth, model, data) ||
            !GetInputScalar(operation, 5, OperandType::INT32, desc.m_PoolHeight, model, data) ||
            !GetInputActivationFunction(operation, 6, activation, model, data))
        {
            return Fail("%s: Operation has invalid inputs", operationName);
        }

        if (Is12OrLaterOperand(*output))
        {
            desc.m_DataLayout = OptionalDataLayout(operation, 7, model, data);
        }

        const armnnUtils::DataLayoutIndexed dataLayout(desc.m_DataLayout);
        const unsigned int inputWidth  = inputInfo.GetShape()[dataLayout.GetWidthIndex()];
        const unsigned int inputHeight = inputInfo.GetShape()[dataLayout.GetHeightIndex()];

        CalcPadding(inputWidth, desc.m_PoolWidth, desc.m_StrideX, desc.m_PadLeft, desc.m_PadRight, scheme);
        CalcPadding(inputHeight, desc.m_PoolHeight, desc.m_StrideY, desc.m_PadTop, desc.m_PadBottom, scheme);
    }

    bool isSupported = false;
    armnn::BackendId setBackend;
    auto validateFunc = [&](const armnn::TensorInfo& outputInfo, bool& isSupported)
    {
        FORWARD_LAYER_SUPPORT_FUNC(__func__,
                                   IsPooling2dSupported,
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

    armnn::IConnectableLayer* pooling2dLayer = data.m_Network->AddPooling2dLayer(desc);
    pooling2dLayer->SetBackendId(setBackend);
    if (!pooling2dLayer)
    {
        return Fail("%s: AddPooling2dLayer failed", __func__);
    }

    input.Connect(pooling2dLayer->GetInputSlot(0));

    if (!isSupported)
    {
        return false;
    }

    return SetupAndTrackLayerOutputSlot(operation, 0, *pooling2dLayer, model,
                                        data, nullptr, validateFunc, activation);
}

bool ConvertReduce(const Operation& operation,
                   const Model& model,
                   ConversionData& data,
                   armnn::ReduceOperation reduceOperation)
{
    armnn::ReduceDescriptor descriptor;
    descriptor.m_ReduceOperation = reduceOperation;

    LayerInputHandle input = ConvertToLayerInputHandle(operation, 0, model, data);
    if (!input.IsValid())
    {
        return Fail("%s: Operation has invalid inputs", __func__);
    }
    const armnn::TensorInfo& inputInfo = input.GetTensorInfo();

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

    // Convert the axis to unsigned int and remove duplicates.
    unsigned int rank = inputInfo.GetNumDimensions();
    std::set<unsigned int> uniqueAxis;
    std::transform(axis.begin(), axis.end(),
                   std::inserter(uniqueAxis, uniqueAxis.begin()),
                   [rank](int i) -> unsigned int { return (i + rank) % rank; });
    descriptor.m_vAxis.assign(uniqueAxis.begin(), uniqueAxis.end());

    // Get the "keep dims" flag.
    if (!GetInputScalar(operation, 2, OperandType::BOOL, descriptor.m_KeepDims, model, data))
    {
        return Fail("%s: Could not read input 2", __func__);
    }

    bool isSupported = false;
    armnn::BackendId setBackend;
    auto validateFunc = [&](const armnn::TensorInfo& outputInfo, bool& isSupported)
    {
        FORWARD_LAYER_SUPPORT_FUNC(__func__,
                                   IsReduceSupported,
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

    armnn::IConnectableLayer* const layer = data.m_Network->AddReduceLayer(descriptor);
    layer->SetBackendId(setBackend);
    assert(layer != nullptr);
    input.Connect(layer->GetInputSlot(0));

    return SetupAndTrackLayerOutputSlot(operation, 0, *layer, model, data, nullptr, validateFunc);
}


bool ConvertToActivation(const Operation& operation,
                         const char* operationName,
                         const armnn::ActivationDescriptor& activationDesc,
                         const Model& model,
                         ConversionData& data)
{
    LayerInputHandle input = ConvertToLayerInputHandle(operation, 0, model, data);
    if (!input.IsValid())
    {
        return Fail("%s: Input 0 is invalid", operationName);
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
                                   activationDesc);
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

    armnn::IConnectableLayer* layer = data.m_Network->AddActivationLayer(activationDesc);
    layer->SetBackendId(setBackend);
    ARMNN_ASSERT(layer != nullptr);
    input.Connect(layer->GetInputSlot(0));

    return SetupAndTrackLayerOutputSlot(operation, 0, *layer, model, data, nullptr, validateFunc);
}

DequantizeResult DequantizeIfRequired(size_t operand_index,
                                      const Operation& operation,
                                      const Model& model,
                                      const ConversionData& data)
{
    const Operand* weightsOperand = GetInputOperand(operation, operand_index, model);
    if (!weightsOperand)
    {
        return { nullptr, 0, armnn::TensorInfo(), DequantizeStatus::INVALID_OPERAND };
    }

    if (IsOperandConstant(*weightsOperand))
    {
        // Weights are already constant
        return { nullptr, 0, armnn::TensorInfo(), DequantizeStatus::NOT_REQUIRED };
    }

    const size_t weightsInputIndex = operation.inputs[operand_index];

    // The weights are a non const tensor, this indicates they might be the output of a dequantize op.
    // Iterate over the nodes and find the previous operation which should be DEQUANTIZE
    for (uint32_t operationIdx = 0; operationIdx < getMainModel(model).operations.size(); ++operationIdx)
    {
        // Search for the DEQUANTIZE op which has the operand with index equal to operandIndex
        const auto& operationIt = getMainModel(model).operations[operationIdx];
        if (operationIt.type != OperationType::DEQUANTIZE)
        {
            continue;
        }

        size_t outOpIndex = weightsInputIndex + 1;
        for (size_t i = 0; outOpIndex != weightsInputIndex && i < operationIt.outputs.size(); ++i)
        {
            outOpIndex = operationIt.outputs[i];
        }

        if (outOpIndex != weightsInputIndex)
        {
            continue;
        }

        const Operand* operand = GetInputOperand(operationIt, 0, model);
        ARMNN_ASSERT(operand);

        if (!IsQSymm8(*operand))
        {
            // Only supporting dequantize from QSYMM8 to FLOAT
            break;
        }

        // Allocate a new buffer for the dequantized data and manually dequantize
        const void* startValue = GetOperandValueReadOnlyAddress(*operand, model, data);
        if (!startValue)
        {
            // Failed to get the operand address
            break;
        }

        const uint8_t* quantizedBuffer = reinterpret_cast<const uint8_t*>(startValue);
        size_t dequantizedBufferLength = operand->location.length;
        const float quantizationScale  = operand->scale;

        auto dequantizedBuffer = std::make_unique<float[]>(dequantizedBufferLength + 1);
        for (size_t i = 0; i < dequantizedBufferLength; ++i)
        {
            float* dstPtr = dequantizedBuffer.get();
            ARMNN_ASSERT(dstPtr);
            *dstPtr++ = quantizedBuffer[i] * quantizationScale;
        }

        // Construct tensor info for dequantized ConstTensor
        armnn::TensorInfo tensorInfo(operand->dimensions.size(),
                                     operand->dimensions.data(),
                                     armnn::DataType::Float32);

        return { std::move(dequantizedBuffer), dequantizedBufferLength * sizeof(float),
                 std::move(tensorInfo),
                 DequantizeStatus::SUCCESS };
    }

    return { nullptr, 0, armnn::TensorInfo() , DequantizeStatus::NOT_REQUIRED};
}

ConstTensorPin DequantizeAndMakeConstTensorPin(const Operation& operation,
                                               const Model& model,
                                               const ConversionData& data,
                                               size_t operandIndex,
                                               bool optional)
{
    DequantizeResult dequantized = DequantizeIfRequired(operandIndex,operation, model, data);

    DequantizeStatus status = std::get<3>(dequantized);
    switch (status)
    {
        case DequantizeStatus::INVALID_OPERAND:
        {
            // return invalid const tensor pin
            return ConstTensorPin();
        }
        case DequantizeStatus::NOT_REQUIRED:
        {
            return ConvertOperationInputToConstTensorPin(
                operation, operandIndex, model, data, g_DontPermute, nullptr, optional);
        }
        case DequantizeStatus::SUCCESS:
        default:
        {
            return ConstTensorPin(
                std::get<2>(dequantized), std::get<0>(dequantized).get(), std::get<1>(dequantized), g_DontPermute);
        }
    }
}

bool GetInputPaddingScheme(const Operation& operation,
                           uint32_t inputIndex,
                           PaddingScheme& outPaddingScheme,
                           const Model& model,
                           const ConversionData& data)
{
    int32_t paddingSchemeAsInt;
    if (!GetInputInt32(operation, inputIndex, paddingSchemeAsInt, model, data))
    {
        return Fail("%s: failed to get padding scheme input value", __func__);
    }

    outPaddingScheme = static_cast<::android::nn::PaddingScheme>(paddingSchemeAsInt);
    return true;
}

const void* GetOperandValueReadOnlyAddress(const Operand& operand,
                                           const Model& model,
                                           const ConversionData& data,
                                           bool optional)
{
    const void* valueStart = nullptr;
    switch (operand.lifetime)
    {
        case OperandLifeTime::CONSTANT_COPY:
        {
            valueStart = model.operandValues.data() + operand.location.offset;
            break;
        }
        case OperandLifeTime::POINTER:
        {
            // Pointer specified in the model
            valueStart = std::get<const void*>(operand.location.pointer);
            break;
        }
        case OperandLifeTime::CONSTANT_REFERENCE:
        {
            // Constant specified via a Memory object
            valueStart = GetMemoryFromPool(operand.location, data.m_MemPools);
            break;
        }
        case OperandLifeTime::NO_VALUE:
        {
            // An optional input tensor with no values is not an error so should not register as a fail
            if (optional)
            {
                valueStart = nullptr;
                break;
            }
            [[fallthrough]];
        }
        default:
        {
            VLOG(DRIVER) << __func__ << ": unsupported/invalid operand lifetime:: " << operand.lifetime;
            valueStart = nullptr;
        }
    }

    return valueStart;
}

bool GetTensorInt32Values(const Operand& operand,
                                 std::vector<int32_t>& outValues,
                                 const Model& model,
                                 const ConversionData& data)
{
    if (operand.type != OperandType::TENSOR_INT32)
    {
        VLOG(DRIVER) << __func__ << ": invalid operand type: " << operand.type;
        return false;
    }

    const void* startAddress = GetOperandValueReadOnlyAddress(operand, model, data);
    if (!startAddress)
    {
        VLOG(DRIVER) << __func__ << ": failed to get operand address " << operand.type;
        return false;
    }

    // Check number of bytes is sensible
    const uint32_t numBytes = operand.location.length;
    if (numBytes % sizeof(int32_t) != 0)
    {
        return Fail("%s: invalid number of bytes: %i, expected to be a multiple of %i",
                    __func__, numBytes, sizeof(int32_t));
    }

    outValues.resize(numBytes / sizeof(int32_t));
    memcpy(outValues.data(), startAddress, numBytes);
    return true;
}

armnn::DataLayout OptionalDataLayout(const Operation& operation,
                                     uint32_t inputIndex,
                                     const Model& model,
                                     ConversionData& data)
{
    const Operand* operand = GetInputOperand(operation, inputIndex, model);
    if (!operand)
    {
        return armnn::DataLayout::NHWC;
    }

    if (!IsBool(*operand))
    {
        return armnn::DataLayout::NHWC;
    }

    const void* valueAddress = GetOperandValueReadOnlyAddress(*operand, model, data);
    if (!valueAddress)
    {
        return armnn::DataLayout::NHWC;
    }

    if (*(static_cast<const bool*>(valueAddress)))
    {
        return armnn::DataLayout::NCHW;
    }
    else
    {
        return armnn::DataLayout::NHWC;
    }
}

armnn::IConnectableLayer* ProcessActivation(const armnn::TensorInfo& tensorInfo,
                                            ActivationFn activation,
                                            armnn::IConnectableLayer* prevLayer,
                                            ConversionData& data)
{
    ARMNN_ASSERT(prevLayer->GetNumOutputSlots() == 1);

    prevLayer->GetOutputSlot(0).SetTensorInfo(tensorInfo);

    armnn::IConnectableLayer* activationLayer = prevLayer;

    if (activation != ActivationFn::kActivationNone)
    {
        armnn::ActivationDescriptor activationDesc;
        switch (activation)
        {
            case ActivationFn::kActivationRelu:
            {
                activationDesc.m_Function = armnn::ActivationFunction::ReLu;
                break;
            }
            case ActivationFn::kActivationRelu1:
            {
                activationDesc.m_Function = armnn::ActivationFunction::BoundedReLu;
                activationDesc.m_A = 1.0f;
                activationDesc.m_B = -1.0f;
                break;
            }
            case ActivationFn::kActivationRelu6:
            {
                activationDesc.m_Function = armnn::ActivationFunction::BoundedReLu;
                activationDesc.m_A = 6.0f;
                break;
            }
            case ActivationFn::kActivationSigmoid:
            {
                activationDesc.m_Function = armnn::ActivationFunction::Sigmoid;
                break;
            }
            case ActivationFn::kActivationTanh:
            {
                activationDesc.m_Function = armnn::ActivationFunction::TanH;
                activationDesc.m_A = 1.0f;
                activationDesc.m_B = 1.0f;
                break;
            }
            default:
            {
                Fail("%s: Invalid activation enum value %i", __func__, activation);
                return nullptr;
            }
        }

        bool isSupported = false;
        armnn::BackendId setBackend;
        FORWARD_LAYER_SUPPORT_FUNC(__func__,
                                   IsActivationSupported,
                                   data.m_Backends,
                                   isSupported,
                                   setBackend,
                                   prevLayer->GetOutputSlot(0).GetTensorInfo(),
                                   tensorInfo,
                                   activationDesc);
        if (!isSupported)
        {
            return nullptr;
        }

        activationLayer = data.m_Network->AddActivationLayer(activationDesc);
        activationLayer->SetBackendId(setBackend);

        prevLayer->GetOutputSlot(0).Connect(activationLayer->GetInputSlot(0));
        activationLayer->GetOutputSlot(0).SetTensorInfo(tensorInfo);
    }

    return activationLayer;
}

bool SetupAndTrackLayerOutputSlot(const Operation& operation,
                                  uint32_t operationOutputIndex,
                                  armnn::IConnectableLayer& layer,
                                  uint32_t layerOutputIndex,
                                  const Model& model,
                                  ConversionData& data,
                                  const armnn::TensorInfo* overrideOutputInfo,
                                  const std::function <void (const armnn::TensorInfo&, bool&)>& validateFunc,
                                  const ActivationFn& activationFunction,
                                  bool inferOutputShapes)
{
    const Operand* outputOperand = GetOutputOperand(operation, operationOutputIndex, model);
    if ((outputOperand == nullptr) || (operationOutputIndex >= layer.GetNumOutputSlots()))
    {
        return false;
    }

    armnn::IOutputSlot& outputSlot = layer.GetOutputSlot(layerOutputIndex);
    if (overrideOutputInfo == nullptr)
    {
        outputSlot.SetTensorInfo(GetTensorInfoForOperand(*outputOperand));
    }
    else
    {
        outputSlot.SetTensorInfo(*overrideOutputInfo);
    }

    bool isSupported = false;
    if (validateFunc && (IsDynamicTensor(outputSlot.GetTensorInfo()) || inferOutputShapes))
    {
        // Type one dynamic tensors require the previous layer's output shape for inference
        for (unsigned int inputSlotIndex = 0; inputSlotIndex < layer.GetNumInputSlots(); ++inputSlotIndex)
        {
            if(!layer.GetInputSlot(inputSlotIndex).GetConnection())
            {
                return false;
            }
        }
        // IsTensorInfoSet will infer the dynamic output shape
        outputSlot.IsTensorInfoSet();
        // Once the shape is inferred we can validate it
        validateFunc(outputSlot.GetTensorInfo(), isSupported);

        if(!isSupported)
        {
            for (unsigned int inputSlotIndex = 0; inputSlotIndex < layer.GetNumInputSlots(); ++inputSlotIndex)
            {
                layer.GetInputSlot(inputSlotIndex).GetConnection()->Disconnect(layer.GetInputSlot(inputSlotIndex));
            }
            return false;
        }
    }

    const uint32_t operandIndex = operation.outputs[operationOutputIndex];

    if (activationFunction != ActivationFn::kActivationNone)
    {
        const armnn::TensorInfo& activationOutputInfo = outputSlot.GetTensorInfo();
        armnn::IConnectableLayer* const endLayer = ProcessActivation(activationOutputInfo, activationFunction,
                                                                     &layer, data);

        if (!endLayer)
        {
            return Fail("%s: ProcessActivation failed", __func__);
        }

        armnn::IOutputSlot& activationOutputSlot = endLayer->GetOutputSlot(layerOutputIndex);
        data.m_OutputSlotForOperand[operandIndex] = &activationOutputSlot;
    }
    else
    {
        data.m_OutputSlotForOperand[operandIndex] = &outputSlot;
    }

    return true;
}

bool IsConnectedToDequantize(armnn::IOutputSlot* ioutputSlot)
{
    VLOG(DRIVER) << "ConversionUtils::IsConnectedToDequantize()";
    if (!ioutputSlot)
    {
        return false;
    }
    VLOG(DRIVER) << "ConversionUtils::IsConnectedToDequantize() ioutputSlot is valid.";
    // Find the connections and layers..
    armnn::IConnectableLayer& owningLayer = ioutputSlot->GetOwningIConnectableLayer();
    if (owningLayer.GetType() == armnn::LayerType::Dequantize)
    {
        VLOG(DRIVER) << "ConversionUtils::IsConnectedToDequantize() connected to Dequantize Layer.";
        armnn::IInputSlot& inputSlot = owningLayer.GetInputSlot(0);
        armnn::IOutputSlot* connection = inputSlot.GetConnection();
        if (connection)
        {
            VLOG(DRIVER) << "ConversionUtils::IsConnectedToDequantize() Dequantize Layer has a connection.";
            armnn::IConnectableLayer& connectedLayer =
                    connection->GetOwningIConnectableLayer();
            if (connectedLayer.GetType() == armnn::LayerType::Constant)
            {
                VLOG(DRIVER) << "ConversionUtils::IsConnectedToDequantize() Dequantize Layer connected to Constant";
                return true;
            }
        }
    }
    return false;
}

} // namespace armnn_driver
