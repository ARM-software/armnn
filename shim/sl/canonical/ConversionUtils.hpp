//
// Copyright © 2022-2023 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include "CanonicalUtils.hpp"

#include <armnn/ArmNN.hpp>
#include <armnn/BackendHelper.hpp>
#include <armnn/utility/Assert.hpp>
#include <armnn/utility/IgnoreUnused.hpp>
#include <armnn/utility/NumericCast.hpp>

#include <armnnUtils/DataLayoutIndexed.hpp>
#include <armnnUtils/Transpose.hpp>

#include <ActivationFunctor.h>
#include <CpuExecutor.h>
#include <OperationsUtils.h>

#include <armnnUtils/FloatingPointComparison.hpp>

#include <log/log.h>
#include <vector>

inline const android::nn::Model::Subgraph& getMainModel(const android::nn::Model& model) { return model.main; }

namespace armnn_driver
{

///
/// Helper classes
///

#include <nnapi/OperandTypes.h>
#include <nnapi/Result.h>
#include <nnapi/TypeUtils.h>
#include <nnapi/Types.h>
#include <nnapi/Validation.h>

using Model                     = ::android::nn::Model;
using Operand                   = ::android::nn::Operand;
using OperandLifeTime           = ::android::nn::Operand::LifeTime;
using OperandType               = ::android::nn::OperandType;
using Operation                 = ::android::nn::Operation;
using OperationType             = ::android::nn::OperationType;
using ErrorStatus               = ::android::nn::ErrorStatus;

struct ConversionData
{
    ConversionData(const std::vector<armnn::BackendId>& backends)
    : m_Backends(backends)
    , m_Network(nullptr, nullptr)
    , m_DynamicInputsEncountered(false)
    {}

    const std::vector<armnn::BackendId>       m_Backends;
    armnn::INetworkPtr                        m_Network;
    std::vector<armnn::IOutputSlot*>          m_OutputSlotForOperand;
    std::vector<::android::nn::RunTimePoolInfo> m_MemPools;
    bool m_DynamicInputsEncountered;
};

class LayerInputHandle
{
public:
    LayerInputHandle();
    LayerInputHandle(bool valid, armnn::IOutputSlot* outputSlot, armnn::TensorInfo tensorInfo);

    bool IsValid() const;

    void Connect(armnn::IInputSlot& inputSlot);

    void Disconnect(armnn::IInputSlot& inputSlot);

    const armnn::TensorInfo& GetTensorInfo() const;

    void SanitizeQuantizationScale(LayerInputHandle& weight, LayerInputHandle& input);

    armnn::IOutputSlot* GetOutputSlot() const;

private:
    armnn::IOutputSlot* m_OutputSlot;
    bool                m_Valid;
    armnn::TensorInfo   m_TensorInfo;
};

class ConstTensorPin
{
public:
    // Creates an invalid tensor pin (can be used to signal errors)
    // The optional flag can be set to indicate the tensor values were missing, but it was otherwise valid
    ConstTensorPin(bool optional = false);

    // @param tensorInfo TensorInfo associated with the tensor.
    // @param valueStart Start address of tensor data. Belongs to one of the memory pools associated with
    // the model being converted.
    // @param numBytes Number of bytes for the tensor data.
    ConstTensorPin(armnn::TensorInfo& tensorInfo, const void* valueStart, uint32_t numBytes,
                   const armnn::PermutationVector& mappings);

    ConstTensorPin(const ConstTensorPin& other) = delete;
    ConstTensorPin(ConstTensorPin&& other)      = default;

    bool IsValid() const;
    bool IsOptional() const;

    const armnn::ConstTensor& GetConstTensor() const;
    const armnn::ConstTensor* GetConstTensorPtr() const;

private:
    armnn::ConstTensor m_ConstTensor;

    // Owned memory for swizzled tensor data, only required if the tensor needed
    // swizzling. Otherwise, @ref m_ConstTensor will reference memory from one of
    // the pools associated with the model being converted.
    std::vector<uint8_t> m_SwizzledTensorData;

    // optional flag to indicate that an invalid tensor pin is not an error, but the optional values were not given
    bool m_Optional;
};

enum class ConversionResult
{
    Success,
    ErrorMappingPools,
    UnsupportedFeature
};

} // namespace armnn_driver

///
/// Utility functions
///

namespace
{
using namespace armnn_driver;

// Convenience function to log the reason for failing to convert a model.
// @return Always returns false (so that it can be used by callers as a quick way to signal an error and return)
template<class... Args>
static bool Fail(const char* formatStr, Args&&... args)
{
    ALOGD(formatStr, std::forward<Args>(args)...);
    return false;
}

// Convenience macro to call an Is*Supported function and log caller name together with reason for lack of support.
// Called as: FORWARD_LAYER_SUPPORT_FUNC(__func__, Is*Supported, backends, a, b, c, d, e)
#define FORWARD_LAYER_SUPPORT_FUNC(funcName, func, backends, supported, setBackend, ...) \
try \
{ \
    for (auto&& backendId : backends) \
    { \
        auto layerSupportObject = armnn::GetILayerSupportByBackendId(backendId); \
        if (layerSupportObject.IsBackendRegistered()) \
        { \
            std::string reasonIfUnsupported; \
            supported = \
                layerSupportObject.func(__VA_ARGS__, armnn::Optional<std::string&>(reasonIfUnsupported)); \
            if (supported) \
            { \
                setBackend = backendId; \
                break; \
            } \
            else \
            { \
                if (reasonIfUnsupported.size() > 0) \
                { \
                    VLOG(DRIVER) << funcName << ": not supported by armnn: " <<  reasonIfUnsupported.c_str(); \
                } \
                else \
                { \
                    VLOG(DRIVER) << funcName << ": not supported by armnn"; \
                } \
            } \
        } \
        else \
        { \
            VLOG(DRIVER) << funcName << ": backend not registered: " << backendId.Get().c_str(); \
        } \
    } \
    if (!supported) \
    { \
        VLOG(DRIVER) << funcName << ": not supported by any specified backend"; \
    } \
} \
catch (const armnn::InvalidArgumentException &e) \
{ \
    throw armnn::InvalidArgumentException(e, "Failed to check layer support", CHECK_LOCATION()); \
}

inline armnn::TensorShape GetTensorShapeForOperand(const Operand& operand)
{
    return armnn::TensorShape(operand.dimensions.size(), operand.dimensions.data());
}

// Support within the 1.3 driver for specific tensor data types
inline bool IsOperandTypeSupportedForTensors(OperandType type)
{
    return type == OperandType::BOOL                           ||
           type == OperandType::TENSOR_BOOL8                   ||
           type == OperandType::TENSOR_FLOAT16                 ||
           type == OperandType::TENSOR_FLOAT32                 ||
           type == OperandType::TENSOR_QUANT8_ASYMM            ||
           type == OperandType::TENSOR_QUANT8_ASYMM_SIGNED     ||
           type == OperandType::TENSOR_QUANT8_SYMM             ||
           type == OperandType::TENSOR_QUANT8_SYMM_PER_CHANNEL ||
           type == OperandType::TENSOR_QUANT16_SYMM            ||
           type == OperandType::TENSOR_INT32;
}

inline bool IsBool(Operand operand)
{
    return operand.type == OperandType::BOOL;
}

inline bool Is12OrLaterOperand(Operand)
{
    return true;
}


template<typename LayerHandleType>
armnn::IConnectableLayer& AddReshapeLayer(armnn::INetwork& network,
                                          LayerHandleType& inputLayer,
                                          armnn::TensorInfo reshapeInfo)
{
    armnn::ReshapeDescriptor reshapeDescriptor;
    reshapeDescriptor.m_TargetShape = reshapeInfo.GetShape();

    armnn::IConnectableLayer* reshapeLayer = network.AddReshapeLayer(reshapeDescriptor);
    ARMNN_ASSERT(reshapeLayer != nullptr);

    // Attach the input layer to the reshape layer
    inputLayer.Connect(reshapeLayer->GetInputSlot(0));
    reshapeLayer->GetOutputSlot(0).SetTensorInfo(reshapeInfo);

    return *reshapeLayer;
}


 armnn::TensorShape FlattenFullyConnectedInput(const armnn::TensorShape& inputShape,
                                               const armnn::TensorShape& weightsShape)
{
    if (inputShape.GetNumDimensions() > 2U)
    {
        unsigned int totalInputElements = inputShape.GetNumElements();
        unsigned int inputSize = weightsShape[1];

        unsigned int batchSize = totalInputElements / inputSize;

        if(totalInputElements % batchSize != 0)
        {
            throw std::runtime_error("Failed to deduce tensor shape");
        }

        return armnn::TensorShape({batchSize, inputSize});
    }
    else
    {
        return inputShape;
    }
}

inline bool VerifyFullyConnectedShapes(const armnn::TensorShape& inputShape,
                                       const armnn::TensorShape& weightsShape,
                                       const armnn::TensorShape& outputShape,
                                       bool  transposeWeightMatrix)
{
    unsigned int dimIdx = transposeWeightMatrix ? 0 : 1;
    return (inputShape[0] == outputShape[0] && weightsShape[dimIdx] == outputShape[1]);
}

bool BroadcastTensor(LayerInputHandle& input0,
                     LayerInputHandle& input1,
                     armnn::IConnectableLayer* startLayer,
                     ConversionData& data)
{
    ARMNN_ASSERT(startLayer != nullptr);

    const armnn::TensorInfo& inputInfo0 = input0.GetTensorInfo();
    const armnn::TensorInfo& inputInfo1 = input1.GetTensorInfo();

    unsigned int inputDimensions0 = inputInfo0.GetNumDimensions();
    unsigned int inputDimensions1 = inputInfo1.GetNumDimensions();

    if (inputDimensions0 == inputDimensions1)
    {
        // The inputs have the same number of dimensions, simply connect them to the given layer as they are
        input0.Connect(startLayer->GetInputSlot(0));
        input1.Connect(startLayer->GetInputSlot(1));

        return true;
    }

    // Since the number of dimensions do not match then we need to add degenerate dimensions
    // to the "smaller" tensor using a reshape, while keeping the order of the inputs.

    unsigned int maxInputDimensions = std::max(inputDimensions0, inputDimensions1);
    unsigned int sizeDifference = std::abs(armnn::numeric_cast<int>(inputDimensions0) -
                                           armnn::numeric_cast<int>(inputDimensions1));

    bool input0IsSmaller = inputDimensions0 < inputDimensions1;
    LayerInputHandle& smallInputHandle = input0IsSmaller ? input0 : input1;
    const armnn::TensorInfo& smallInfo = smallInputHandle.GetTensorInfo();

    const armnn::TensorShape& smallShape = smallInfo.GetShape();
    std::vector<unsigned int> reshapedDimensions(maxInputDimensions, 1);
    for (unsigned int i = sizeDifference; i < maxInputDimensions; i++)
    {
        reshapedDimensions[i] = smallShape[i - sizeDifference];
    }

    armnn::TensorInfo reshapedInfo = smallInfo;
    reshapedInfo.SetShape(armnn::TensorShape{ armnn::numeric_cast<unsigned int>(reshapedDimensions.size()),
                                              reshapedDimensions.data() });

    // RehsapeDescriptor that is ignored in the IsReshapeSupported function
    armnn::ReshapeDescriptor reshapeDescriptor;

    bool isSupported = false;
    armnn::BackendId setBackend;
    FORWARD_LAYER_SUPPORT_FUNC(__func__,
                               IsReshapeSupported,
                               data.m_Backends,
                               isSupported,
                               setBackend,
                               smallInfo,
                               reshapedInfo,
                               reshapeDescriptor);
    if (!isSupported)
    {
        return false;
    }

    ARMNN_ASSERT(data.m_Network != nullptr);
    armnn::IConnectableLayer& reshapeLayer = AddReshapeLayer(*data.m_Network, smallInputHandle, reshapedInfo);
    reshapeLayer.SetBackendId(setBackend);

    if (input0IsSmaller)
    {
        // Input0 is the "smaller" tensor, connect the reshape layer as follows:
        //
        //  Input0 Input1
        //     |     |
        //  Reshape  |
        //      \   /
        //    StartLayer

        reshapeLayer.GetOutputSlot(0).Connect(startLayer->GetInputSlot(0));
        input1.Connect(startLayer->GetInputSlot(1));
    }
    else
    {
        // Input1 is the "smaller" tensor, connect the reshape layer as follows:
        //
        //  Input0 Input1
        //     |     |
        //     |  Reshape
        //      \   /
        //    StartLayer

        input0.Connect(startLayer->GetInputSlot(0));
        reshapeLayer.GetOutputSlot(0).Connect(startLayer->GetInputSlot(1));
    }

    return true;
}

void CalcPadding(uint32_t input,
                 uint32_t kernel,
                 uint32_t stride,
                 uint32_t& outPadHead,
                 uint32_t& outPadTail,
                 PaddingScheme scheme)
{
    int32_t padHead;
    int32_t padTail;
    calculateExplicitPadding(input, stride, kernel, scheme, &padHead, &padTail);
    outPadHead = armnn::numeric_cast<uint32_t>(padHead);
    outPadTail = armnn::numeric_cast<uint32_t>(padTail);
}

void CalcPadding(uint32_t input, uint32_t kernel, uint32_t stride, uint32_t dilation, uint32_t& outPadHead,
                 uint32_t& outPadTail, ::android::nn::PaddingScheme scheme)
{
    int32_t padHead;
    int32_t padTail;
    calculateExplicitPadding(input, stride, dilation, kernel, scheme, &padHead, &padTail);
    outPadHead = armnn::numeric_cast<uint32_t>(padHead);
    outPadTail = armnn::numeric_cast<uint32_t>(padTail);
}

inline void CalcPaddingTransposeConv(uint32_t output, uint32_t kernel, int32_t stride, int32_t& outPadHead,
                              int32_t& outPadTail, ::android::nn::PaddingScheme scheme)
{
    calculateExplicitPaddingTransposeConv(output, stride, kernel, scheme, &outPadHead, &outPadTail);
}

Shape GetOperandShape(const Operand& operand)
{
    Shape shape;
    shape.type = OperandType(operand.type);
    shape.dimensions = operand.dimensions;
    shape.scale = operand.scale;
    shape.offset = operand.zeroPoint;
    return shape;
}


// ArmNN requires the bias scale to be equal to the product of the weight and input scales, which is also
// what AndroidNN requires. However for some of the AndroidNN tests the values don't exactly match so
// we accept some tolerance. We don't want ArmNN itself to accept these inconsistencies as it is up to the
// user (us, in this case) to ensure they match.
void SanitizeBiasQuantizationScale(armnn::TensorInfo& biasInfo,
                                   const armnn::TensorInfo& weightInfo,
                                   const armnn::TensorInfo& inputInfo)
{
    if (weightInfo.HasPerAxisQuantization())
    {
        // NOTE: Bias scale is always set to 0 for per-axis quantization and
        // it needs to be calculated: scale[i] = input_scale * weight_scale[i]
        auto UpdateBiasScaleValue = [&inputInfo](float biasScale) -> float
        {
            return biasScale * inputInfo.GetQuantizationScale();
        };

        std::vector<float> biasScales(weightInfo.GetQuantizationScales());
        std::transform(biasScales.begin(), biasScales.end(), biasScales.begin(), UpdateBiasScaleValue);

        biasInfo.SetQuantizationScales(biasScales);
        // bias is expected to be a 1d tensor, set qdim=0
        biasInfo.SetQuantizationDim(0);

        VLOG(DRIVER) << "Bias quantization params have been updated for per-axis quantization";
    }
    else
    {
        const float expectedBiasScale = weightInfo.GetQuantizationScale() * inputInfo.GetQuantizationScale();
        if (biasInfo.GetQuantizationScale() != expectedBiasScale)
        {
            if (armnnUtils::within_percentage_tolerance(biasInfo.GetQuantizationScale(), expectedBiasScale, 1.0f))
            {
                VLOG(DRIVER) << "Bias quantization scale has been modified to match input * weights";
                biasInfo.SetQuantizationScale(expectedBiasScale);
            }
        }
    }
}

// 4D Tensor Permutations
const armnn::PermutationVector IdentityPermutation4D({ 0U, 1U, 2U, 3U });
const armnn::PermutationVector IdentityPermutation3D({ 0U, 1U, 2U });
const armnn::PermutationVector SwapDim2And3({ 0U, 1U, 3U, 2U });

// 3D Permutation Vectors
const armnn::PermutationVector RotateTensorLeft({ 1U, 2U, 0U });
const armnn::PermutationVector RotateTensorRight({ 2U, 0U, 1U });

template<typename OSlot>
armnn::IConnectableLayer& AddTransposeLayer(armnn::INetwork& network, OSlot& input,
                                            const armnn::PermutationVector& mappings)
{
    // Add swizzle layer
    armnn::IConnectableLayer* const layer = network.AddTransposeLayer(mappings);

    ARMNN_ASSERT(layer != nullptr);

    // Connect input to swizzle layer
    input.Connect(layer->GetInputSlot(0));

    // Setup swizzled output
    const armnn::TensorInfo outInfo = armnnUtils::TransposeTensorShape(input.GetTensorInfo(), mappings);
    layer->GetOutputSlot(0).SetTensorInfo(outInfo);

    return *layer;
}

bool ValidateConcatOutputShape(const std::vector<armnn::TensorShape> & inputShapes,
                               const armnn::TensorShape & outputShape,
                               uint32_t concatDim)
{
    // Validate the output shape is correct given the input shapes (which have just been validated)
    unsigned int numDimensions = inputShapes[0].GetNumDimensions();
    if (outputShape.GetNumDimensions() != numDimensions)
    {
        return Fail("%s: Output shape has wrong number of dimensions", __func__);
    }

    unsigned int outputSizeAlongConcatenatedDimension = 0;
    for (unsigned int i = 0; i < inputShapes.size(); i++)
    {
        outputSizeAlongConcatenatedDimension += inputShapes[i][concatDim];
    }

    for (unsigned int i = 0; i < numDimensions; ++i)
    {
        if (i == concatDim)
        {
            if (outputShape[i] != outputSizeAlongConcatenatedDimension)
            {
                return Fail(
                        "%s: Invalid output shape for dimension %d (%d != %d)",
                        __func__,
                        i,
                        outputShape[i],
                        outputSizeAlongConcatenatedDimension);
            }
        }
        else
        {
            if (outputShape[i] != inputShapes[0][i])
            {
                return Fail("%s: Invalid output shape", __func__);
            }
        }
    }

    return true;
}

inline bool RequiresReshape(armnn::TensorShape & inputShape)
{
    return inputShape.GetNumDimensions() < 3;
}

inline void SwizzleInputs(armnn::INetwork& network,
                   std::vector<LayerInputHandle>& inputs,
                   std::vector<armnn::TensorShape>& inputShapes,
                   const armnn::PermutationVector& mapping,
                   std::vector<armnn::BackendId>& setBackends)
{
    if (!mapping.IsEqual(IdentityPermutation4D))
    {
        size_t nInputs = inputs.size();
        for (size_t i=0; i<nInputs; ++i)
        {
            // add swizzle layer
            armnn::IConnectableLayer& swizzleLayer = AddTransposeLayer(network, inputs[i], mapping);
            swizzleLayer.SetBackendId(setBackends[i]);
            auto& outputSlot = swizzleLayer.GetOutputSlot(0);
            auto& outputInfo = outputSlot.GetTensorInfo();
            // replace inputs with the swizzled ones
            inputs[i] = LayerInputHandle(true, &outputSlot, outputInfo);
            inputShapes[i] = inputs[i].GetTensorInfo().GetShape();
        }
    }
}

bool TransposeInputTensors(ConversionData& data,
                          std::vector<LayerInputHandle>& inputs,
                          std::vector<armnn::TensorShape>& inputShapes,
                          const armnn::PermutationVector& mapping)
{
    // If we have a IdentityPermutation4D or IdentityPermutation3D then we are not permuting
    if (!mapping.IsEqual(IdentityPermutation4D) && !mapping.IsEqual(IdentityPermutation3D))
    {
        std::vector<armnn::BackendId> setBackendsVec;
        armnn::TensorInfo outputTransposeInfo;
        size_t nInputs = inputs.size();
        for (size_t i=0; i<nInputs; ++i)
        {
            // check permute layer
            armnn::TransposeDescriptor transposeDesc;
            transposeDesc.m_DimMappings = mapping;
            outputTransposeInfo = armnnUtils::TransposeTensorShape(inputs[i].GetTensorInfo(), mapping);

            bool isSupported = false;
            armnn::BackendId setBackend;
            FORWARD_LAYER_SUPPORT_FUNC(__func__,
                                       IsTransposeSupported,
                                       data.m_Backends,
                                       isSupported,
                                       setBackend,
                                       inputs[i].GetTensorInfo(),
                                       outputTransposeInfo,
                                       transposeDesc);
            setBackendsVec.push_back(setBackend);
            if (!isSupported)
            {
                return false;
            }

        }
        SwizzleInputs(*data.m_Network, inputs, inputShapes, mapping, setBackendsVec);
    }
    return true;
}

bool CreateConcatPermutationParameters(const unsigned int numberOfDimensions,
                                       int32_t & concatDimension,
                                       std::pair<armnn::PermutationVector, armnn::PermutationVector> & permutationPair)
{
    bool needPermute = false;
    ARMNN_ASSERT(numberOfDimensions >= 3);

    // ArmNN uses Compute Library subtensors to perform concatenation
    // This only works when concatenating along dimension 0, 1 or 3 for a 4-D tensor,
    // or along dimension 0 or 2 for a 3-D tensor.
    if (numberOfDimensions == 4 && concatDimension == 2)
    {
        concatDimension = 3;
        permutationPair = std::make_pair(SwapDim2And3, SwapDim2And3);
        needPermute = true;
    }
    else if (numberOfDimensions == 3 && concatDimension == 1)
    {
        concatDimension = 0;
        permutationPair = std::make_pair(RotateTensorLeft, RotateTensorRight);
        needPermute = true;
    }
    // If the tensor is 3-D and the concat dimension is 2 then we don't need to permute but we do need to change the
    // permutation identity to only have 3 dimensions
    else if (numberOfDimensions == 3 && concatDimension == 2)
    {
        permutationPair = std::make_pair(IdentityPermutation3D, IdentityPermutation3D);
    }
    return needPermute;
}

} // anonymous namespace

namespace armnn_driver
{
using namespace android::nn;

//// Creates an ArmNN activation layer and connects it to the given layer, if the
//// passed in AndroidNN activation function requires so.
//// @return The end layer of the sequence of layers built for the given AndroidNN
//// activation function or nullptr if an error occurred (e.g. unsupported activation).
//// Note that the end layer matches the input layer if no activation is required
//// (the sequence of layers has length 1).
armnn::IConnectableLayer* ProcessActivation(const armnn::TensorInfo& tensorInfo,
                                            ActivationFn activation,
                                            armnn::IConnectableLayer* prevLayer,
                                            ConversionData& data);


inline const Operand* GetInputOperand(const Operation& operation,
                                      uint32_t inputIndex,
                                      const Model& model,
                                      bool failOnIndexOutOfBounds = true)
{
    if (inputIndex >= operation.inputs.size())
    {
        if (failOnIndexOutOfBounds)
        {
            Fail("%s: invalid input index: %i out of %i", __func__, inputIndex, operation.inputs.size());
        }
        return nullptr;
    }

    // Model should have been validated beforehand
    ARMNN_ASSERT(operation.inputs[inputIndex] < getMainModel(model).operands.size());
    return &getMainModel(model).operands[operation.inputs[inputIndex]];
}

inline const Operand* GetOutputOperand(const Operation& operation,
                                       uint32_t outputIndex,
                                       const Model& model)
{
    if (outputIndex >= operation.outputs.size())
    {
        Fail("%s: invalid output index: %i out of %i", __func__, outputIndex, operation.outputs.size());
        return nullptr;
    }

    // Model should have been validated beforehand
    ARMNN_ASSERT(operation.outputs[outputIndex] < getMainModel(model).operands.size());

    return &getMainModel(model).operands[operation.outputs[outputIndex]];
}

const void* GetOperandValueReadOnlyAddress(const Operand& operand,
                                           const Model& model,
                                           const ConversionData& data,
                                           bool optional = false);

inline bool GetOperandType(const Operation& operation,
                           uint32_t inputIndex,
                           const Model& model,
                           OperandType& type)
{
    const Operand* operand = GetInputOperand(operation, inputIndex, model);
    if (!operand)
    {
        return Fail("%s: invalid input operand at index %i", __func__, inputIndex);
    }

    type = operand->type;
    return true;
}

inline bool IsOperandConstant(const Operand& operand)
{
    OperandLifeTime lifetime = operand.lifetime;

    return lifetime == OperandLifeTime::CONSTANT_COPY ||
           lifetime == OperandLifeTime::CONSTANT_REFERENCE ||
           lifetime == OperandLifeTime::POINTER ||
           lifetime == OperandLifeTime::NO_VALUE;
}

bool IsWeightsValid(const Operation& operation, uint32_t inputIndex, const Model& model);

ConstTensorPin ConvertOperandToConstTensorPin(const Operand& operand,
                                              const Model& model,
                                              const ConversionData& data,
                                              const armnn::PermutationVector& dimensionMappings = g_DontPermute,
                                              const armnn::TensorShape* overrideTensorShape = nullptr,
                                              bool optional = false,
                                              const armnn::DataType* overrideDataType = nullptr);

inline ConstTensorPin ConvertOperationInputToConstTensorPin(
        const Operation& operation,
        uint32_t inputIndex,
        const Model& model,
        const ConversionData& data,
        const armnn::PermutationVector& dimensionMappings = g_DontPermute,
        const armnn::TensorShape* overrideTensorShape = nullptr,
        bool optional = false)
{
    const Operand* operand = GetInputOperand(operation, inputIndex, model);
    if (!operand)
    {
        Fail("%s: failed to get input operand: index=%u", __func__, inputIndex);
        return ConstTensorPin();
    }
    return ConvertOperandToConstTensorPin(*operand,
                                          model,
                                          data,
                                          dimensionMappings,
                                          overrideTensorShape,
                                          optional);
}

template <typename OutputType>
bool GetInputScalar(const Operation& operation,
                    uint32_t inputIndex,
                    OperandType type,
                    OutputType& outValue,
                    const Model& model,
                    const ConversionData& data,
                    bool optional = false)
{
    const Operand* operand = GetInputOperand(operation, inputIndex, model);
    if (!optional && !operand)
    {
        return Fail("%s: invalid input operand at index %i", __func__, inputIndex);
    }

    if (!optional && operand->type != type)
    {
        VLOG(DRIVER) << __func__ << ": unexpected operand type: " << operand->type << " should be: " << type;
        return false;
    }

    if (!optional && operand->location.length != sizeof(OutputType))
    {
        return Fail("%s: incorrect operand location length: %i (should be %i)",
                    __func__, operand->location.length, sizeof(OutputType));
    }

    const void* valueAddress = GetOperandValueReadOnlyAddress(*operand, model, data);
    if (!optional && !valueAddress)
    {
        return Fail("%s: failed to get address for operand", __func__);
    }

    if(!optional)
    {
        outValue = *(static_cast<const OutputType*>(valueAddress));
    }

    return true;
}

inline bool GetInputInt32(const Operation& operation,
                          uint32_t inputIndex,
                          int32_t& outValue,
                          const Model& model,
                          const ConversionData& data)
{
    return GetInputScalar(operation, inputIndex, OperandType::INT32, outValue, model, data);
}

inline bool GetInputFloat32(const Operation& operation,
                            uint32_t inputIndex,
                            float& outValue,
                            const Model& model,
                            const ConversionData& data)
{
    return GetInputScalar(operation, inputIndex, OperandType::FLOAT32, outValue, model, data);
}

inline bool GetInputActivationFunctionImpl(const Operation& operation,
                                           uint32_t inputIndex,
                                           OperandType type,
                                           ActivationFn& outActivationFunction,
                                           const Model& model,
                                           const ConversionData& data)
{
    if (type != OperandType::INT32 && type != OperandType::TENSOR_INT32)
    {
        VLOG(DRIVER) << __func__ << ": unexpected operand type: " << type
                     << " should be OperandType::INT32 or OperandType::TENSOR_INT32";
        return false;
    }

    int32_t activationFunctionAsInt;
    if (!GetInputScalar(operation, inputIndex, type, activationFunctionAsInt, model, data))
    {
        return Fail("%s: failed to get activation input value", __func__);
    }
    outActivationFunction = static_cast<ActivationFn>(activationFunctionAsInt);
    return true;
}

inline bool GetInputActivationFunction(const Operation& operation,
                                       uint32_t inputIndex,
                                       ActivationFn& outActivationFunction,
                                       const Model& model,
                                       const ConversionData& data)
{
    return GetInputActivationFunctionImpl(operation,
                                          inputIndex,
                                          OperandType::INT32,
                                          outActivationFunction,
                                          model,
                                          data);
}

inline bool GetInputActivationFunctionFromTensor(const Operation& operation,
                                                 uint32_t inputIndex,
                                                 ActivationFn& outActivationFunction,
                                                 const Model& model,
                                                 const ConversionData& data)
{
    // This only accepts a 1-D tensor of size 1
    return GetInputActivationFunctionImpl(operation,
                                          inputIndex,
                                          OperandType::INT32,
                                          outActivationFunction,
                                          model,
                                          data);
}


inline bool GetOptionalInputActivation(const Operation& operation,
                                       uint32_t inputIndex,
                                       ActivationFn& activationFunction,
                                       const Model& model,
                                       const ConversionData& data)
{
    if (operation.inputs.size() <= inputIndex)
    {
        activationFunction = ActivationFn::kActivationNone;
    }
    else
    {
        if (!GetInputActivationFunction(operation, inputIndex, activationFunction, model, data))
        {
            return Fail("%s: Operation has invalid inputs", __func__);
        }
    }
    return true;
}

template<typename ConvolutionDescriptor>
bool GetOptionalConvolutionDilationParams(const Operation& operation,
                                          uint32_t dilationXIndex,
                                          ConvolutionDescriptor& descriptor,
                                          const Model& model,
                                          const ConversionData& data)
{
    bool success = true;
    if (operation.inputs.size() >= dilationXIndex + 2)
    {
        success &= GetInputScalar(operation,
                                  dilationXIndex,
                                  OperandType::INT32,
                                  descriptor.m_DilationX,
                                  model,
                                  data);
        success &= GetInputScalar(operation,
                                  dilationXIndex + 1,
                                  OperandType::INT32,
                                  descriptor.m_DilationY,
                                  model,
                                  data);
    }

    return success;
}

inline bool GetOptionalBool(const Operation& operation,
                            uint32_t inputIndex,
                            const Model& model,
                            const ConversionData& data)
{
    const Operand* operand = GetInputOperand(operation, inputIndex, model);
    if (!operand)
    {
        return false;
    }

    if (!IsBool(*operand))
    {
        return false;
    }

    const void* valueAddress = GetOperandValueReadOnlyAddress(*operand, model, data);
    if (!valueAddress)
    {
        return false;
    }

    return *(static_cast<const bool*>(valueAddress));
}

bool GetTensorInt32Values(const Operand& operand,
                                 std::vector<int32_t>& outValues,
                                 const Model& model,
                                 const ConversionData& data);

bool GetInputPaddingScheme(const Operation& operation,
                           uint32_t inputIndex,
                           PaddingScheme& outPaddingScheme,
                           const Model& model,
                           const ConversionData& data);

LayerInputHandle ConvertToLayerInputHandle(const Operation& operation,
                                           uint32_t inputIndex,
                                           const Model& model,
                                           ConversionData& data,
                                           const armnn::PermutationVector& dimensionMappings = g_DontPermute,
                                           const LayerInputHandle* inputHandle = nullptr);

bool SetupAndTrackLayerOutputSlot(const Operation& operation,
                                  uint32_t operationOutputIndex,
                                  armnn::IConnectableLayer& layer,
                                  uint32_t layerOutputIndex,
                                  const Model& model,
                                  ConversionData& data,
                                  const armnn::TensorInfo* overrideOutputInfo = nullptr,
                                  const std::function <void (const armnn::TensorInfo&, bool&)>& validateFunc = nullptr,
                                  const ActivationFn& activationFunction = ActivationFn::kActivationNone,
                                  bool inferOutputShapes = false);

armnn::DataLayout OptionalDataLayout(const Operation& operation,
                                     uint32_t inputIndex,
                                     const Model& model,
                                     ConversionData& data);

inline bool SetupAndTrackLayerOutputSlot(
        const Operation& operation,
        uint32_t outputIndex,
        armnn::IConnectableLayer& layer,
        const Model& model,
        ConversionData& data,
        const armnn::TensorInfo* overrideOutputInfo = nullptr,
        const std::function <void (const armnn::TensorInfo&, bool&)>& validateFunc = nullptr,
        const ActivationFn& activationFunction = ActivationFn::kActivationNone)
{
    return SetupAndTrackLayerOutputSlot(operation,
                                        outputIndex,
                                        layer,
                                        outputIndex,
                                        model,
                                        data,
                                        overrideOutputInfo,
                                        validateFunc,
                                        activationFunction);
}

bool ConvertToActivation(const Operation& operation,
                         const char* operationName,
                         const armnn::ActivationDescriptor& activationDesc,
                         const Model& model,
                         ConversionData& data);

bool ConvertPaddings(const Operation& operation,
                     const Model& model,
                     ConversionData& data,
                     unsigned int rank,
                     armnn::PadDescriptor& padDescriptor);
bool ConvertReduce(const Operation& operation,
                   const Model& model,
                   ConversionData& data,
                   armnn::ReduceOperation reduceOperation);

bool ConvertPooling2d(const Operation& operation,
                      const char* operationName,
                      armnn::PoolingAlgorithm poolType,
                      const Model& model,
                      ConversionData& data);

inline bool IsQSymm8(const Operand& operand)
{
    return operand.type == OperandType::TENSOR_QUANT8_SYMM;
}

enum class DequantizeStatus
{
    SUCCESS,
    NOT_REQUIRED,
    INVALID_OPERAND
};

using DequantizeResult = std::tuple<std::unique_ptr<float[]>, size_t, armnn::TensorInfo, DequantizeStatus>;

DequantizeResult DequantizeIfRequired(size_t operand_index,
                                      const Operation& operation,
                                      const Model& model,
                                      const ConversionData& data);

ConstTensorPin DequantizeAndMakeConstTensorPin(const Operation& operation,
                                               const Model& model,
                                               const ConversionData& data,
                                               size_t operandIndex,
                                               bool optional = false);

bool IsConnectedToDequantize(armnn::IOutputSlot* ioutputSlot);

} // namespace armnn_driver
