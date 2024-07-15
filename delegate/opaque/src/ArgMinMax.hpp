//
// Copyright © 2023-2024 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include <OpaqueDelegateUtils.hpp>

#include <tensorflow/lite/builtin_ops.h>
#include <tensorflow/lite/c/builtin_op_data.h>
#include <tensorflow/lite/c/common.h>
#include <tensorflow/lite/minimal_logging.h>

namespace armnnOpaqueDelegate
{
TfLiteStatus VisitArgMinMaxOperator(DelegateData& delegateData,
                                    TfLiteOpaqueContext* tfLiteContext,
                                    TfLiteOpaqueNode* tfLiteNode,
                                    int nodeIndex,
                                    int32_t argMinMaxOperatorCode)
{
    TF_LITE_ENSURE_STATUS(ValidateNumInputs(tfLiteContext, tfLiteNode, 2, nodeIndex));
    TF_LITE_ENSURE_STATUS(ValidateNumOutputs(tfLiteContext, tfLiteNode, 1, nodeIndex));

    // Gather input indices and use to get input tensor.
    auto numInputs = TfLiteOpaqueNodeNumberOfInputs(tfLiteNode);
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
    if (!IsValid(tfLiteContext, tfLiteInputTensor, argMinMaxOperatorCode, nodeIndex))
    {
        return kTfLiteError;
    }

    // Use input indices to get filter tensor.
    const TfLiteOpaqueTensor* tfLiteAxisTensor = TfLiteOpaqueContextGetOpaqueTensor(tfLiteContext, inputTensors[1]);
    if(!IsValid(tfLiteAxisTensor))
    {
        TF_LITE_OPAQUE_MAYBE_KERNEL_LOG(
                tfLiteContext,
                "TfLiteArmnnOpaqueDelegate: Invalid filter tensor in operator #%d node #%d: ",
                argMinMaxOperatorCode, nodeIndex);
        return kTfLiteError;
    }

    // Gather output indices and use to get output tensors.
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
    if (!IsValid(tfLiteContext, tfLiteOutputTensor, argMinMaxOperatorCode, nodeIndex))
    {
        return kTfLiteError;
    }

    const armnn::TensorInfo& inputTensorInfo = GetTensorInfoForTfLiteOpaqueTensor(tfLiteInputTensor);
    const armnn::TensorInfo& outputTensorInfo = GetTensorInfoForTfLiteOpaqueTensor(tfLiteOutputTensor, true);

    if(outputTensorInfo.GetShape().GetDimensionality() == armnn::Dimensionality::NotSpecified)
    {
        TF_LITE_OPAQUE_MAYBE_KERNEL_LOG(
            tfLiteContext,
            "TfLiteArmnnDelegate: NotSpecified Dimensionality is not supported in operator #%d node #%d: ",
            argMinMaxOperatorCode, nodeIndex);
        return kTfLiteError;
    }

    // Get const axis value from model and set it to descriptor.
    if (!IsValid(tfLiteContext, tfLiteAxisTensor, argMinMaxOperatorCode, nodeIndex))
    {
        return kTfLiteError;
    }

    armnn::ArgMinMaxDescriptor desc;
    auto* axisData = static_cast<int*>(TfLiteOpaqueTensorData(tfLiteAxisTensor));
    // Get the axis value from the input tensor
    switch (TfLiteOpaqueTensorType(tfLiteAxisTensor))
    {
        case kTfLiteInt32:
        case kTfLiteInt64:
            desc.m_Axis = axisData[0];
            break;
        default:
            TF_LITE_OPAQUE_MAYBE_KERNEL_LOG(
                tfLiteContext,
                "TfLiteArmnnOpaqueDelegate: Axis value data type is not supported in operator #%d node #%d: ",
                argMinMaxOperatorCode, nodeIndex);
            return kTfLiteError;
    }

    // If output_type is int32 then set Signed32 else Signed64. Default type is Signed64.
    if (argMinMaxOperatorCode == kTfLiteBuiltinArgMax)
    {
        desc.m_Function = armnn::ArgMinMaxFunction::Max;
        auto* argMaxParameters = reinterpret_cast<TfLiteArgMaxParams*>(TfLiteOpaqueNodeGetBuiltinData(tfLiteNode));
        if (argMaxParameters->output_type != kTfLiteInt32 && argMaxParameters->output_type != kTfLiteInt64)
        {
            TF_LITE_OPAQUE_MAYBE_KERNEL_LOG(
                tfLiteContext,
                "TfLiteArmnnOpaqueDelegate: output_type data type is not supported in operator #%d node #%d: ",
                argMinMaxOperatorCode, nodeIndex);
            return kTfLiteError;
        }
    }
    else
    {
        desc.m_Function = armnn::ArgMinMaxFunction::Min;
        auto* argMinParameters = reinterpret_cast<TfLiteArgMinParams*>(TfLiteOpaqueNodeGetBuiltinData(tfLiteNode));
        if (argMinParameters->output_type != kTfLiteInt32 && argMinParameters->output_type != kTfLiteInt64)
        {
            TF_LITE_OPAQUE_MAYBE_KERNEL_LOG(
                    tfLiteContext,
                    "TfLiteArmnnOpaqueDelegate: output_type data type is not supported in operator #%d node #%d: ",
                    argMinMaxOperatorCode, nodeIndex);
            return kTfLiteError;
        }
    }

    bool isSupported = false;
    armnn::BackendId setBackend;
    auto validateFunc = [&](const armnn::TensorInfo& outInfo, bool& isSupported)
    {
        FORWARD_LAYER_OPAQUE_SUPPORT_FUNC("ARGMINMAX",
                                           tfLiteContext,
                                           IsArgMinMaxSupported,
                                           delegateData.m_Backends,
                                           isSupported,
                                           setBackend,
                                           inputTensorInfo,
                                           outInfo,
                                           desc);
    };

    if (!delegateData.m_Network)
    {
        validateFunc(outputTensorInfo, isSupported);
        return isSupported ? kTfLiteOk : kTfLiteError;
    }

    // Add an ArgMinMax layer
    auto layerName = GetName(desc.m_Function, nodeIndex);
    armnn::IConnectableLayer* layer = delegateData.m_Network->AddArgMinMaxLayer(desc, layerName.c_str());
    layer->SetBackendId(setBackend);
    ARMNN_ASSERT(layer != nullptr);

    armnn::IOutputSlot& outputSlot = layer->GetOutputSlot(0);
    outputSlot.SetTensorInfo(outputTensorInfo);

    // try to connect the Constant Inputs if there are any
    if (ProcessInputs(layer, delegateData, tfLiteContext, tfLiteNode, nodeIndex) != kTfLiteOk)
    {
        return kTfLiteError;
    }

    // Connect
    return Connect(layer, tfLiteContext, tfLiteNode, delegateData);
}

}