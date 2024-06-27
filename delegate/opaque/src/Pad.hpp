//
// Copyright © 2023-2024 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include <OpaqueDelegateUtils.hpp>

namespace armnnOpaqueDelegate
{

TfLiteStatus VisitPadOperator(DelegateData& delegateData,
                              TfLiteOpaqueContext* tfLiteContext,
                              TfLiteOpaqueNode* tfLiteNode,
                              int nodeIndex,
                              int32_t tfLitePadOperatorCode)
{

    switch(tfLitePadOperatorCode)
    {
        case kTfLiteBuiltinMirrorPad:
        case kTfLiteBuiltinPad:
            TF_LITE_ENSURE_STATUS(ValidateNumInputs(tfLiteContext, tfLiteNode, 2, nodeIndex));
            break;
        case kTfLiteBuiltinPadv2:
            TF_LITE_ENSURE_STATUS(ValidateNumInputs(tfLiteContext, tfLiteNode, 3, nodeIndex));
            break;
        default:
            return kTfLiteError;
    }

    // Inputs
    int numInputs = 0;
    const int* inputTensors;

    if (TfLiteOpaqueNodeInputs(tfLiteNode, &inputTensors, &numInputs) != kTfLiteOk)
    {
        TF_LITE_OPAQUE_MAYBE_KERNEL_LOG(
                tfLiteContext,
                "TfLiteArmnnOpaqueDelegate: Unable to gather input tensor indices from node #%d: ",
                nodeIndex);
        return kTfLiteError;
    }

    const TfLiteOpaqueTensor* tfLiteInputTensor = TfLiteOpaqueContextGetOpaqueTensor(tfLiteContext, inputTensors[0]);
    if (!IsValid(tfLiteContext, tfLiteInputTensor, tfLitePadOperatorCode, nodeIndex))
    {
        TF_LITE_OPAQUE_MAYBE_KERNEL_LOG(tfLiteContext,
                                        "TfLiteArmnnOpaqueDelegate: Invalid input tensor at node #%d: ",
                                        nodeIndex);
        return kTfLiteError;
    }

    const TfLiteOpaqueTensor* tfLitePaddingTensor = TfLiteOpaqueContextGetOpaqueTensor(tfLiteContext, inputTensors[1]);
    if (!IsValid(tfLiteContext, tfLitePaddingTensor, tfLitePadOperatorCode, nodeIndex))
    {
        TF_LITE_OPAQUE_MAYBE_KERNEL_LOG(tfLiteContext,
                                        "TfLiteArmnnOpaqueDelegate: Invalid padding tensor at node #%d: ",
                                        nodeIndex);
        return kTfLiteError;
    }

    // Output
    TF_LITE_ENSURE_STATUS(ValidateNumOutputs(tfLiteContext, tfLiteNode, 1, nodeIndex));

    int numOutputs = 0;
    const int* outputTensors;
    if (TfLiteOpaqueNodeOutputs(tfLiteNode, &outputTensors, &numOutputs) != kTfLiteOk)
    {
        TF_LITE_OPAQUE_MAYBE_KERNEL_LOG(
                tfLiteContext,
                "TfLiteArmnnOpaqueDelegate: Unable to gather output tensor indices from node #%d: ",
                nodeIndex);
        return kTfLiteError;
    }

    const TfLiteOpaqueTensor* tfLiteOutputTensor = TfLiteOpaqueContextGetOpaqueTensor(tfLiteContext, outputTensors[0]);
    if (!IsValid(tfLiteContext, tfLiteOutputTensor, tfLitePadOperatorCode, nodeIndex))
    {
        TF_LITE_OPAQUE_MAYBE_KERNEL_LOG(tfLiteContext,
                                        "TfLiteArmnnOpaqueDelegate: Invalid output tensor at node #%d: ", nodeIndex);
        return kTfLiteError;
    }

    const armnn::TensorInfo& inputTensorInfo = GetTensorInfoForTfLiteOpaqueTensor(tfLiteInputTensor);
    const armnn::TensorInfo& paddingTensorInfo = GetTensorInfoForTfLiteOpaqueTensor(tfLitePaddingTensor);
    const armnn::TensorInfo& outputTensorInfo = GetTensorInfoForTfLiteOpaqueTensor(tfLiteOutputTensor, true);

    armnn::PadDescriptor descriptor;

    // Get the padding size from the input tensor 1
    size_t step = 2;
    if(paddingTensorInfo.GetDataType() == armnn::DataType::Signed64)
    {
        auto* paddingData = static_cast<int64_t*>(TfLiteOpaqueTensorData(tfLitePaddingTensor));
        for (uint16_t i = 0; i < paddingTensorInfo.GetNumElements() / step; ++i)
        {
            descriptor.m_PadList.emplace_back(paddingData[i * step], paddingData[i * step + 1]);
        }
    }
    else
    {
        auto* paddingData = static_cast<int32_t*>(TfLiteOpaqueTensorData(tfLitePaddingTensor));
        for (uint16_t i = 0; i < paddingTensorInfo.GetNumElements() / step; ++i)
        {
            descriptor.m_PadList.emplace_back(paddingData[i * step], paddingData[i * step + 1]);
        }
    }

    // Get the padding value from input tensor 2
    if (tfLitePadOperatorCode == kTfLiteBuiltinPad && inputTensorInfo.IsQuantized())
    {
        descriptor.m_PadValue = inputTensorInfo.GetQuantizationOffset();
    }
    else if (tfLitePadOperatorCode == kTfLiteBuiltinPadv2)
    {
        const TfLiteOpaqueTensor* tfLitePaddingValue = TfLiteOpaqueContextGetOpaqueTensor(tfLiteContext,
                                                                                          inputTensors[2]);
        if (!IsValid(tfLiteContext, tfLitePaddingValue, tfLitePadOperatorCode, nodeIndex))
        {
            TF_LITE_OPAQUE_MAYBE_KERNEL_LOG(tfLiteContext,
                                            "TfLiteArmnnOpaqueDelegate: Invalid padding value at node #%d: ",
                                            nodeIndex);
            return kTfLiteError;
        }

        armnn::TensorInfo paddingValueTensorInfo = GetTensorInfoForTfLiteOpaqueTensor(tfLitePaddingValue);
        if (paddingValueTensorInfo.GetNumElements() != 1)
        {
            TF_LITE_OPAQUE_MAYBE_KERNEL_LOG(
                    tfLiteContext,
                    "TfLiteArmnnOpaqueDelegate: Multiple padding value are not supported in operator #%d node #%d: ",
                    tfLitePadOperatorCode, nodeIndex);
            return kTfLiteError;
        }

        // Get the padding value from the input tensor
        switch (TfLiteOpaqueTensorType(tfLitePaddingValue))
        {
            case kTfLiteFloat32:
                descriptor.m_PadValue = *static_cast<float*>(TfLiteOpaqueTensorData(tfLitePaddingValue));
                break;
            case kTfLiteInt32:
                descriptor.m_PadValue = *static_cast<int32_t*>(TfLiteOpaqueTensorData(tfLitePaddingValue));
                break;
            case kTfLiteUInt8:
                descriptor.m_PadValue = *static_cast<uint8_t*>(TfLiteOpaqueTensorData(tfLitePaddingValue));
                break;
            case kTfLiteInt8:
                descriptor.m_PadValue = *static_cast<int8_t*>(TfLiteOpaqueTensorData(tfLitePaddingValue));
                break;
            default:
                TF_LITE_OPAQUE_MAYBE_KERNEL_LOG(
                        tfLiteContext,
                        "TfLiteArmnnOpaqueDelegate: Padding value datatype is not supported in operator #%d node #%d: ",
                        tfLitePadOperatorCode, nodeIndex);
                return kTfLiteError;
        }
    }
    else if (tfLitePadOperatorCode == kTfLiteBuiltinMirrorPad)
    {
        auto* options = reinterpret_cast<TfLiteMirrorPaddingParams*>(TfLiteOpaqueNodeGetBuiltinData(tfLiteNode));

        if (options->mode == TfLiteMirrorPaddingMode::kTfLiteMirrorPaddingReflect)
        {
            descriptor.m_PaddingMode = armnn::PaddingMode::Reflect;
        }
        else if (options->mode == TfLiteMirrorPaddingMode::kTfLiteMirrorPaddingSymmetric)
        {
            descriptor.m_PaddingMode = armnn::PaddingMode::Symmetric;
        }
        else
        {
            TF_LITE_OPAQUE_MAYBE_KERNEL_LOG(
                    tfLiteContext,
                    "TfLiteArmnnOpaqueDelegate: PaddingMode must be either REFLECT or SYMMETRIC "
                    "in operator #%d node #%d: ",
                    tfLitePadOperatorCode, nodeIndex);
        }

        // If padding mode is Reflect then both paddings must be no greater than inputShape(i) - 1.
        // If padding mode is Symmetric then both paddings must be no greater than inputShape(i).
        auto inputShape = inputTensorInfo.GetShape();
        auto padList = descriptor.m_PadList;

        const auto isReflect = static_cast<unsigned int>(descriptor.m_PaddingMode == armnn::PaddingMode::Reflect);
        for(unsigned int i = 0; i < padList.size(); ++i)
        {
            if(padList.at(i).first > (inputShape[i] - isReflect) ||
               padList.at(i).second > (inputShape[i] - isReflect))
            {
                TF_LITE_OPAQUE_MAYBE_KERNEL_LOG(
                        tfLiteContext,
                        "TfLiteArmnnOpaqueDelegate: Padding values must be less (Reflect) or "
                        "equal (Symmetric) to the dimension size in operator #%d node #%d: ",
                        tfLitePadOperatorCode, nodeIndex);
            }
        }
    }

    armnn::BackendId setBackend;
    bool isSupported = false;
    auto validateFunc = [&](const armnn::TensorInfo& outputTensorInfo)
    {
        FORWARD_LAYER_OPAQUE_SUPPORT_FUNC("PAD",
                                          tfLiteContext,
                                          IsPadSupported,
                                          delegateData.m_Backends,
                                          isSupported,
                                          setBackend,
                                          inputTensorInfo,
                                          outputTensorInfo,
                                          descriptor);
    };

    if (!delegateData.m_Network)
    {
        validateFunc(outputTensorInfo);
        return isSupported ? kTfLiteOk : kTfLiteError;
    }

    auto layerName = GetName(armnn::LayerType::Pad, nodeIndex);
    armnn::IConnectableLayer* padLayer = delegateData.m_Network->AddPadLayer(descriptor, layerName.c_str());
    padLayer->SetBackendId(setBackend);
    ARMNN_ASSERT(padLayer != nullptr);

    armnn::IOutputSlot& outputSlot = padLayer->GetOutputSlot(0);
    outputSlot.SetTensorInfo(outputTensorInfo);

    if (ProcessInputs(padLayer, delegateData, tfLiteContext, tfLiteNode, nodeIndex) != kTfLiteOk)
    {
        return kTfLiteError;
    }

    return Connect(padLayer, tfLiteContext, tfLiteNode, delegateData);
}

} // namespace armnnOpaqueDelegate