//
// Copyright © 2020 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include <armnn/ArmNN.hpp>
#include <armnn/BackendHelper.hpp>
#include <armnn/utility/Assert.hpp>
#include <armnn/utility/NumericCast.hpp>

#include <armnnUtils/Permute.hpp>

#include <tensorflow/lite/builtin_ops.h>
#include <tensorflow/lite/c/builtin_op_data.h>
#include <tensorflow/lite/c/common.h>
#include <tensorflow/lite/minimal_logging.h>

#include "tensorflow/lite/kernels/kernel_util.h"

namespace
{

// Macro to call an Is<layer_name>Supported function and log caller name together with reason for lack of support
#define FORWARD_LAYER_SUPPORT_FUNC(funcName, tfLiteContext, func, backends, supported, ...) \
try \
{ \
    for (auto&& backendId : backends) \
    { \
        auto layerSupportObject = armnn::GetILayerSupportByBackendId(backendId); \
        if (layerSupportObject) \
        { \
            std::string reasonIfUnsupported; \
            supported = \
                layerSupportObject->func(__VA_ARGS__, armnn::Optional<std::string&>(reasonIfUnsupported)); \
            if (supported) \
            { \
                break; \
            } \
            else \
            { \
                if (reasonIfUnsupported.size() > 0) \
                { \
                    TF_LITE_KERNEL_LOG( \
                        tfLiteContext, "%s: not supported by armnn: %s", funcName, reasonIfUnsupported.c_str()); \
                } \
                else \
                { \
                    TF_LITE_KERNEL_LOG(tfLiteContext, "%s: not supported by armnn", funcName); \
                } \
            } \
        } \
        else \
        { \
            TF_LITE_KERNEL_LOG(tfLiteContext, "%s: backend not registered: %s", funcName, backendId.Get().c_str()); \
        } \
    } \
    if (!supported) \
    { \
        TF_LITE_KERNEL_LOG(tfLiteContext, "%s: not supported by any specified backend", funcName); \
    } \
} \
catch (const armnn::InvalidArgumentException &e) \
{ \
    throw armnn::InvalidArgumentException(e, "Failed to check layer support", CHECK_LOCATION()); \
}

TfLiteStatus ValidateNumInputs(TfLiteContext* tfLiteContext,
                               TfLiteNode* tfLiteNode,
                               const unsigned int expectedSize,
                               int nodeIndex)
{
    auto numInputs = tfLiteNode->inputs->size;
    if (static_cast<unsigned int >(numInputs) != expectedSize)
    {
        TF_LITE_MAYBE_KERNEL_LOG(
            tfLiteContext, "TfLiteArmnnDelegate: Unexpected number of inputs (%d != %d) in node #%d",
            numInputs, expectedSize, nodeIndex);
        return kTfLiteError;
    }
    return kTfLiteOk;
}

TfLiteStatus ValidateNumOutputs(TfLiteContext* tfLiteContext,
                                TfLiteNode* tfLiteNode,
                                const unsigned int expectedSize,
                                int nodeIndex)
{
    auto numOutputs = tfLiteNode->outputs->size;
    if (static_cast<unsigned int >(numOutputs) != expectedSize)
    {
        TF_LITE_MAYBE_KERNEL_LOG(
            tfLiteContext, "TfLiteArmnnDelegate: Unexpected number of outputs (%d != %d) in node #%d",
            numOutputs, expectedSize, nodeIndex);
        return kTfLiteError;
    }
    return kTfLiteOk;
}

bool IsValid(const TfLiteTensor* tfLiteTensor)
{
    return tfLiteTensor == nullptr ? false : true;
}

uint32_t NonNegative(int32_t value, int nodeIndex)
{
    if (value < 0)
    {
        throw armnn::Exception("TfLiteArmnnDelegate: Non-negative value in node " + nodeIndex);
    }
    else
    {
        return static_cast<uint32_t>(value);
    }
}

bool IsDynamicTensor(const TfLiteTensor& tfLiteTensor)
{
    auto tensorAllocationType = tfLiteTensor.allocation_type;
    if (tensorAllocationType == kTfLiteDynamic)
    {
        return true;
    }
    return false;
}

bool IsAffineQuantization(const TfLiteTensor& tfLiteTensor)
{
    auto quantizationInfo = tfLiteTensor.quantization;
    if (quantizationInfo.type == kTfLiteAffineQuantization)
    {
        return true;
    }
    return false;
}

TfLiteStatus Connect(armnn::IConnectableLayer* layer,
                     TfLiteNode* tfLiteNode,
                     armnnDelegate::DelegateData& data)
{
    ARMNN_ASSERT(static_cast<unsigned int >(tfLiteNode->outputs->size) == layer->GetNumOutputSlots());

    // Connect the input slots
    for (unsigned int inputIndex = 0; inputIndex < layer->GetNumInputSlots(); ++inputIndex)
    {
        if (data.m_OutputSlotForNode[tfLiteNode->inputs->data[inputIndex]] != nullptr)
        {
            data.m_OutputSlotForNode[tfLiteNode->inputs->data[inputIndex]]->Connect(layer->GetInputSlot(inputIndex));
        }
    }

    // Prepare output slots
    for (unsigned int outputIndex = 0; outputIndex < layer->GetNumOutputSlots(); ++outputIndex)
    {
        armnn::IOutputSlot& outputSlot = layer->GetOutputSlot(outputIndex);
        data.m_OutputSlotForNode[static_cast<unsigned long>(tfLiteNode->outputs->data[outputIndex])] = &outputSlot;
    }

    return kTfLiteOk;
}

armnn::IConnectableLayer* BroadcastTensor(const armnn::TensorInfo& inputInfo0,
                                          const armnn::TensorInfo& inputInfo1,
                                          armnn::IConnectableLayer* startLayer,
                                          TfLiteContext* tfLiteContext,
                                          TfLiteNode* tfLiteNode,
                                          armnnDelegate::DelegateData& delegateData)
{
    unsigned int inputDimensions0 = inputInfo0.GetNumDimensions();
    unsigned int inputDimensions1 = inputInfo1.GetNumDimensions();

    if (inputDimensions0 == inputDimensions1)
    {
        auto status = Connect(startLayer, tfLiteNode, delegateData);
        return status == kTfLiteOk ? startLayer : nullptr;
    }

    unsigned int biggerInputDimensions = std::max(inputDimensions0, inputDimensions1);
    unsigned int dimDifference = static_cast<unsigned int>(std::abs(armnn::numeric_cast<int>(inputDimensions0) -
                                                                    armnn::numeric_cast<int>(inputDimensions1)));

    bool input0IsSmaller = inputDimensions0 < inputDimensions1;
    const armnn::TensorInfo& smallInfo = input0IsSmaller ? inputInfo0 : inputInfo1;
    const armnn::TensorShape& smallShape = smallInfo.GetShape();

    std::vector<unsigned int> reshapedDimensions(biggerInputDimensions, 1);
    for (unsigned int i = dimDifference; i < biggerInputDimensions; ++i)
    {
        reshapedDimensions[i] = smallShape[i - dimDifference];
    }

    armnn::TensorInfo reshapedInfo = smallInfo;
    reshapedInfo.SetShape(armnn::TensorShape{ armnn::numeric_cast<unsigned int>(reshapedDimensions.size()),
                                              reshapedDimensions.data() });

    armnn::ReshapeDescriptor reshapeDescriptor;
    bool isSupported = false;
    FORWARD_LAYER_SUPPORT_FUNC(__func__,
                               tfLiteContext,
                               IsReshapeSupported,
                               delegateData.m_Backends,
                               isSupported,
                               smallInfo,
                               reshapedInfo,
                               reshapeDescriptor);
    if (!isSupported)
    {
        return nullptr;
    }

    ARMNN_ASSERT(delegateData.m_Network != nullptr);
    // Add Reshape layer
    reshapeDescriptor.m_TargetShape = reshapedInfo.GetShape();

    armnn::IConnectableLayer* reshapeLayer = delegateData.m_Network->AddReshapeLayer(reshapeDescriptor);
    ARMNN_ASSERT(reshapeLayer != nullptr);
    reshapeLayer->GetOutputSlot(0).SetTensorInfo(reshapedInfo);

    if (input0IsSmaller)
    {
        delegateData.m_OutputSlotForNode[static_cast<unsigned long>(tfLiteNode->inputs->data[0])]
            ->Connect(reshapeLayer->GetInputSlot(0));
        reshapeLayer->GetOutputSlot(0).Connect(startLayer->GetInputSlot(0));
        delegateData.m_OutputSlotForNode[static_cast<unsigned long>(tfLiteNode->inputs->data[1])]
            ->Connect(startLayer->GetInputSlot(1));
    }
    else
    {
        delegateData.m_OutputSlotForNode[static_cast<unsigned long>(tfLiteNode->inputs->data[1])]
            ->Connect(reshapeLayer->GetInputSlot(0));
        reshapeLayer->GetOutputSlot(0).Connect(startLayer->GetInputSlot(1));
        delegateData.m_OutputSlotForNode[static_cast<unsigned long>(tfLiteNode->inputs->data[0])]
            ->Connect(startLayer->GetInputSlot(0));
    }

    // Prepare output slots
    for (unsigned int outputIndex = 0; outputIndex < startLayer->GetNumOutputSlots(); ++outputIndex)
    {
        armnn::IOutputSlot& outputSlot = startLayer->GetOutputSlot(outputIndex);
        delegateData.m_OutputSlotForNode
            [static_cast<unsigned long>(tfLiteNode->outputs->data[outputIndex])] = &outputSlot;
    }

    return reshapeLayer;
}

TfLiteStatus FusedActivation(TfLiteContext* tfLiteContext,
                             TfLiteNode* tfLiteNode,
                             TfLiteFusedActivation activationType,
                             armnn::IConnectableLayer* prevLayer,
                             unsigned int outputSlotIndex,
                             armnnDelegate::DelegateData& data)
{

    const armnn::TensorInfo& activationOutputInfo = prevLayer->GetOutputSlot(outputSlotIndex).GetTensorInfo();

    armnn::ActivationDescriptor activationDesc;

    switch (activationType)
    {
        case kTfLiteActNone:
        {
            // No Activation
            return kTfLiteOk;
        }
        case kTfLiteActRelu:
        {
            activationDesc.m_Function = armnn::ActivationFunction::ReLu;
            break;
        }
        case kTfLiteActRelu1:
        {
            activationDesc.m_Function = armnn::ActivationFunction::BoundedReLu;
            activationDesc.m_A = 1.0f;
            activationDesc.m_B = -1.0f;
            break;
        }
        case kTfLiteActRelu6:
        {
            activationDesc.m_Function = armnn::ActivationFunction::BoundedReLu;
            activationDesc.m_A = 6.0f;
            activationDesc.m_B = 0.0f;
            break;
        }
        case kTfLiteActSigmoid:
        {
            activationDesc.m_Function = armnn::ActivationFunction::Sigmoid;
            break;
        }
        case kTfLiteActTanh:
        {
            activationDesc.m_Function = armnn::ActivationFunction::TanH;
            activationDesc.m_A = 1.0f;
            activationDesc.m_B = 1.0f;
            break;
        }
        default:
            return kTfLiteError;
    }

    bool isSupported = false;
    FORWARD_LAYER_SUPPORT_FUNC(__func__,
                               tfLiteContext,
                               IsActivationSupported,
                               data.m_Backends,
                               isSupported,
                               prevLayer->GetOutputSlot(0).GetTensorInfo(),
                               activationOutputInfo,
                               activationDesc);
    if (!isSupported)
    {
        return kTfLiteError;
    }
    armnn::IConnectableLayer* activationLayer = data.m_Network->AddActivationLayer(activationDesc);

    ARMNN_ASSERT(activationLayer != nullptr);
    activationLayer->GetOutputSlot(0).SetTensorInfo(activationOutputInfo);

    // Connect and prepare output slots
    for (unsigned int outputIndex = 0; outputIndex < activationLayer->GetNumOutputSlots(); ++outputIndex)
    {
        data.m_OutputSlotForNode[static_cast<unsigned long>(
                tfLiteNode->outputs->data[outputIndex])]->Connect(activationLayer->GetInputSlot(0));
        armnn::IOutputSlot& outputSlot = activationLayer->GetOutputSlot(outputIndex);
        data.m_OutputSlotForNode[static_cast<unsigned long>(
                tfLiteNode->outputs->data[outputIndex])] = &outputSlot;
    }
    return kTfLiteOk;
}

armnn::DataType GetDataType(const TfLiteTensor& tfLiteTensor)
{
    switch (tfLiteTensor.type)
    {
        case kTfLiteBool:
            return armnn::DataType::Boolean;
        case kTfLiteFloat32:
            return armnn::DataType::Float32;
        case kTfLiteFloat16:
            return armnn::DataType::Float16;
        case kTfLiteUInt8:
            return armnn::DataType::QAsymmU8;
        case kTfLiteInt8:
        {
            auto quantizationInfo = tfLiteTensor.quantization;
            if (quantizationInfo.type == kTfLiteAffineQuantization)
            {
                auto* quantization =
                    reinterpret_cast<TfLiteAffineQuantization*>(tfLiteTensor.quantization.params);
                if (quantization->zero_point != nullptr && quantization->zero_point->size == 1)
                {
                    return armnn::DataType::QAsymmS8;
                }
                else
                {
                    return armnn::DataType::QSymmS8;
                }
            }
            else
            {
                return armnn::DataType::QAsymmS8;
            }
        }
        case kTfLiteInt16:
            return armnn::DataType::QSymmS16;
        case kTfLiteInt32:
            return armnn::DataType::Signed32;
        default:
            throw armnn::Exception(&"TfLiteArmnnDelegate: Unsupported data type: " [ tfLiteTensor.type]);
    }
}

armnn::TensorInfo GetTensorInfoForTfLiteTensor(const TfLiteTensor& tfLiteTensor,
                                               const armnn::PermutationVector& dimensionMappings = {0, 1, 2, 3})
{
    armnn::DataType type = GetDataType(tfLiteTensor);
    armnn::TensorInfo ret;
    auto tensorDimensionSize = tfLiteTensor.dims->size;
    if (tensorDimensionSize == 0)
    {
        if(tflite::IsConstantTensor(&tfLiteTensor))
        {
            std::vector<unsigned int> safeShape = { 1 };
            bool dimensionsSpecificity[1] = { true };
            armnn::TensorShape tensorShape(armnn::numeric_cast<unsigned int>(safeShape.size()),
                                           safeShape.data(),
                                           dimensionsSpecificity);
            ret = armnn::TensorInfo(tensorShape, type);
        }
        else
        {
            armnn::TensorShape tensorShape(armnn::Dimensionality::NotSpecified);
            ret = armnn::TensorInfo(tensorShape, type);
        }
    }
    else
    {
        std::vector<unsigned int> tensorDims(static_cast<unsigned int>(tensorDimensionSize));
        bool dimensionsSpecificity[5] = { true, true, true, true, true };
        for (unsigned int i = 0; i < static_cast<unsigned int>(tensorDimensionSize); ++i) {
            auto dim = tfLiteTensor.dims->data[i];
            if (dim == 0)
            {
                dimensionsSpecificity[i] = false;
            }
            tensorDims[i] = static_cast<unsigned int>(dim);
        }
        armnn::TensorShape tensorShape(static_cast<unsigned int>(tensorDimensionSize),
                                       tensorDims.data(),
                                       dimensionsSpecificity);
        ret = armnn::TensorInfo(tensorShape, type);
    }

    auto quantizationInfo = tfLiteTensor.quantization;
    if (quantizationInfo.type == kTfLiteAffineQuantization)
    {
        // get per-channel quantization parameters
        const auto* affineQuantization =
            reinterpret_cast<TfLiteAffineQuantization*>(tfLiteTensor.quantization.params);
        if (affineQuantization->scale->size > 1)
        {
            std::vector<float> quantizationScales;
            for (unsigned int i = 1; i < static_cast<unsigned int>(affineQuantization->scale->size); ++i)
            {
                quantizationScales.push_back(affineQuantization->scale->data[i]);
            }
            ret.SetQuantizationScales(quantizationScales);
            ret.SetQuantizationDim(dimensionMappings[armnn::numeric_cast<unsigned int>(
                affineQuantization->quantized_dimension)]);
        }
        else
        {
            ret.SetQuantizationScale(affineQuantization->scale->data[0]);
            ret.SetQuantizationOffset(affineQuantization->zero_point->data[0]);
        }
    }
    else
    {
        auto quantizationParameters = tfLiteTensor.params;
        ret.SetQuantizationScale(quantizationParameters.scale);
        ret.SetQuantizationOffset(quantizationParameters.zero_point);
    }

    return ret;
}

armnn::ConstTensor CreateConstTensor(const TfLiteTensor* tfLiteTensor,
                                     armnn::TensorInfo& tensorInfo,
                                     armnn::Optional<armnn::PermutationVector&> permutationVector,
                                     void* permutationData = nullptr)
{
    if (tfLiteTensor->allocation_type != kTfLiteMmapRo)
    {
        throw armnn::Exception("TfLiteArmnnDelegate: Not constant allocation type: " + tfLiteTensor->allocation_type);
    }

    if (permutationVector.has_value() && permutationVector.value().GetSize() > 0 && permutationData != nullptr)
    {
        armnnUtils::Permute(armnnUtils::Permuted(tensorInfo.GetShape(), permutationVector.value()),
                            permutationVector.value(),
                            tfLiteTensor->data.data,
                            permutationData,
                            armnn::GetDataTypeSize(tensorInfo.GetDataType()));

        return armnn::ConstTensor(armnnUtils::Permuted(tensorInfo, permutationVector.value()), permutationData);
    }
    else
    {
        return armnn::ConstTensor(tensorInfo, tfLiteTensor->data.data);
    }
}

void CalcPadding(uint32_t inputSize,
                 uint32_t filterSize,
                 uint32_t stride,
                 uint32_t dilation,
                 uint32_t& paddingFront,
                 uint32_t& paddingBack,
                 TfLitePadding padding)
{
    paddingFront = 0;
    paddingBack = 0;
    if (padding == kTfLitePaddingSame)
    {
        uint32_t outputSize = (inputSize + stride - 1) / stride;
        uint32_t dilatedSize = filterSize + (dilation - 1) * (filterSize - 1);
        uint32_t temp = (outputSize - 1) * stride + dilatedSize;
        if (temp > inputSize)
        {
            paddingFront = (temp - inputSize) / 2;
            paddingBack = (temp - inputSize) - paddingFront;
        }
    }
}

TfLiteStatus ConnectConstant(armnn::IConnectableLayer* layer,
                             armnn::TensorInfo& constTensorInfo,
                             TfLiteContext* tfLiteContext,
                             const TfLiteTensor& tfLiteTensor,
                             armnnDelegate::DelegateData& data,
                             unsigned int slotIndex)
{
    bool isSupported = false;
    FORWARD_LAYER_SUPPORT_FUNC(__func__,
                               tfLiteContext,
                               IsConstantSupported,
                               data.m_Backends,
                               isSupported,
                               constTensorInfo);
    if (!isSupported)
    {
        return kTfLiteError;
    }

    auto constantInput = CreateConstTensor(&tfLiteTensor,
                                           constTensorInfo,
                                           armnn::Optional<armnn::PermutationVector&>());
    armnn::IConnectableLayer* constantLayer = data.m_Network->AddConstantLayer(constantInput);
    armnn::IOutputSlot& outputSlot = constantLayer->GetOutputSlot(0);
    outputSlot.SetTensorInfo(constTensorInfo);

    data.m_OutputSlotForNode[static_cast<unsigned long>(slotIndex)] = &outputSlot;

    return kTfLiteOk;
}

} // namespace anonymous
