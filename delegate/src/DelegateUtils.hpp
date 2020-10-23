//
// Copyright © 2020 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include <armnn/ArmNN.hpp>
#include <armnn/BackendHelper.hpp>
#include <armnn/utility/Assert.hpp>

#include <tensorflow/lite/builtin_ops.h>
#include <tensorflow/lite/c/builtin_op_data.h>
#include <tensorflow/lite/c/common.h>
#include <tensorflow/lite/minimal_logging.h>

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
    if (numInputs != expectedSize)
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
    if (numOutputs != expectedSize)
    {
        TF_LITE_MAYBE_KERNEL_LOG(
            tfLiteContext, "TfLiteArmnnDelegate: Unexpected number of outputs (%d != %d) in node #%d",
            numOutputs, expectedSize, nodeIndex);
        return kTfLiteError;
    }
    return kTfLiteOk;
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

armnn::TensorInfo GetTensorInfoForTfLiteTensor(const TfLiteTensor& tfLiteTensor)
{
    armnn::DataType type;
    switch (tfLiteTensor.type)
    {
        case kTfLiteBool:
            type = armnn::DataType::Boolean;
            break;
        case kTfLiteFloat32:
            type = armnn::DataType::Float32;
            break;
        case kTfLiteFloat16:
            type = armnn::DataType::Float16;
            break;
        case kTfLiteUInt8:
            type = armnn::DataType::QAsymmU8;
            break;
        case kTfLiteInt8:
            type = armnn::DataType::QSymmS8;
            break;
        case kTfLiteInt16:
            type = armnn::DataType::QSymmS16;
            break;
        case kTfLiteInt32:
            type = armnn::DataType::Signed32;
            break;
        default:
            throw armnn::Exception("TfLiteArmnnDelegate: Unsupported data type: " + tfLiteTensor.type);
    }

    armnn::TensorInfo ret;
    auto tensorDimensionSize = tfLiteTensor.dims->size;
    if (tensorDimensionSize == 0)
    {
        armnn::TensorShape tensorShape(armnn::Dimensionality::NotSpecified);
        ret = armnn::TensorInfo(tensorShape, type);
    }
    else
    {
        std::vector<unsigned int> tensorDims(tensorDimensionSize);
        bool dimensionsSpecificity[5] = { true, true, true, true, true };
        for (unsigned int i = 0; i < tensorDimensionSize; ++i) {
            auto dim = tfLiteTensor.dims->data[i];
            if (dim == 0)
            {
                dimensionsSpecificity[i] = false;
            }
            tensorDims[i] = dim;
        }
        armnn::TensorShape tensorShape(tensorDimensionSize, tensorDims.data(), dimensionsSpecificity);
        ret = armnn::TensorInfo(tensorShape, type);
    }

    auto quantizationInfo = tfLiteTensor.quantization;
    if (quantizationInfo.type == kTfLiteAffineQuantization)
    {
        // get per-channel quantization parameters
        const auto* affineQuantization =
            reinterpret_cast<TfLiteAffineQuantization*>(tfLiteTensor.quantization.params);
        std::vector<float> quantizationScales;
        for (unsigned int i = 1; i < affineQuantization->scale->size; ++i)
        {
            quantizationScales.push_back(affineQuantization->scale->data[i]);
        }
        ret.SetQuantizationScales(quantizationScales);
        ret.SetQuantizationDim(armnn::MakeOptional<unsigned int>(affineQuantization->quantized_dimension));
    }
    else
    {
        auto quantizationParameters = tfLiteTensor.params;
        ret.SetQuantizationScale(quantizationParameters.scale);
        ret.SetQuantizationOffset(quantizationParameters.zero_point);
    }

    return ret;
}

TfLiteStatus Connect(armnn::IConnectableLayer& layer,
                     TfLiteNode* tfLiteNode,
                     armnnDelegate::DelegateData& data)
{
    ARMNN_ASSERT(tfLiteNode->inputs->size  == layer.GetNumInputSlots());
    ARMNN_ASSERT(tfLiteNode->outputs->size == layer.GetNumOutputSlots());

    // connect the input slots
    for (unsigned int inputIndex = 0; inputIndex < layer.GetNumInputSlots(); ++inputIndex)
    {
        data.m_OutputSlotForNode[tfLiteNode->inputs->data[inputIndex]]->Connect(layer.GetInputSlot(inputIndex));
    }

    // prepare output slots
    for (unsigned int outputIndex = 0; outputIndex < layer.GetNumOutputSlots(); ++outputIndex)
    {
        armnn::IOutputSlot& outputSlot = layer.GetOutputSlot(outputIndex);
        data.m_OutputSlotForNode[tfLiteNode->outputs->data[outputIndex]] = &outputSlot;
    }
    return kTfLiteOk;
}

} // namespace anonymous
