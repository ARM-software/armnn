//
// Copyright © 2023 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//
#pragma once

#include <OpaqueDelegateUtils.hpp>

namespace armnnOpaqueDelegate
{

TfLiteStatus VisitReduceOperator(DelegateData& delegateData,
                                 TfLiteOpaqueContext* tfLiteContext,
                                 TfLiteOpaqueNode* tfLiteNode,
                                 int nodeIndex,
                                 int32_t reduceOperatorCode)
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

    const TfLiteOpaqueTensor* tfLiteInputTensor =
            TfLiteOpaqueContextGetOpaqueTensor(tfLiteContext, inputTensors[0]);
    if (!IsValid(tfLiteContext, tfLiteInputTensor, reduceOperatorCode, nodeIndex))
    {
        return kTfLiteError;
    }

    const TfLiteOpaqueTensor* tfLiteAxisTensor =
            TfLiteOpaqueContextGetOpaqueTensor(tfLiteContext, inputTensors[1]);
    if (!IsValid(tfLiteContext, tfLiteAxisTensor, reduceOperatorCode, nodeIndex))
    {
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

    const TfLiteOpaqueTensor* tfLiteOutputTensor =
            TfLiteOpaqueContextGetOpaqueTensor(tfLiteContext, outputTensors[0]);
    if (!IsValid(tfLiteContext, tfLiteOutputTensor, reduceOperatorCode, nodeIndex))
    {
        return kTfLiteError;
    }

    const armnn::TensorInfo& inputTensorInfo  = GetTensorInfoForTfLiteOpaqueTensor(tfLiteInputTensor);
    const armnn::TensorInfo& outputTensorInfo = GetTensorInfoForTfLiteOpaqueTensor(tfLiteOutputTensor, true);

    // Get const axis value from model and set it to descriptor.
    const armnn::TensorInfo& axisTensorInfo =   GetTensorInfoForTfLiteOpaqueTensor(tfLiteAxisTensor);
    auto* axisTensorData = static_cast<int*>(TfLiteOpaqueTensorData(tfLiteAxisTensor));

    std::vector<int32_t> axis;
    // Add axis data to vector to be converter to unsigned int and assigned to descriptor axis.
    if (axisTensorData != nullptr)
    {
        for (unsigned int i = 0; i < axisTensorInfo.GetNumElements(); ++i)
        {
            axis.emplace_back(axisTensorData[i]);
        }
    }
    else
    {
        for (unsigned int i = 0; i < inputTensorInfo.GetNumDimensions(); ++i)
        {
            axis.push_back(i);
        }
    }

    // Convert the axis to unsigned int and remove duplicates.
    unsigned int rank = inputTensorInfo.GetNumDimensions();
    std::set<unsigned int> uniqueAxis;
    std::transform(axis.begin(),
                   axis.end(),
                   std::inserter(uniqueAxis, uniqueAxis.begin()),
                   [rank](int i)->unsigned int{ return (i + rank) % rank; });

    armnn::ReduceDescriptor desc;
    desc.m_vAxis.assign(uniqueAxis.begin(), uniqueAxis.end());

    auto* reducerParameters = reinterpret_cast<TfLiteReducerParams*>(TfLiteOpaqueNodeGetBuiltinData(tfLiteNode));
    desc.m_KeepDims = reducerParameters->keep_dims;
    if (reduceOperatorCode == kTfLiteBuiltinReduceMax)
    {
        desc.m_ReduceOperation = armnn::ReduceOperation::Max;
    }
    else if (reduceOperatorCode == kTfLiteBuiltinReduceMin)
    {
        desc.m_ReduceOperation = armnn::ReduceOperation::Min;
    }
    else if (reduceOperatorCode == kTfLiteBuiltinSum)
    {
        desc.m_ReduceOperation = armnn::ReduceOperation::Sum;
    }
    else if (reduceOperatorCode == kTfLiteBuiltinReduceProd)
    {
        desc.m_ReduceOperation = armnn::ReduceOperation::Prod;
    }
    else
    {
        TF_LITE_OPAQUE_MAYBE_KERNEL_LOG(
                tfLiteContext,
                "TfLiteArmnnOpaqueDelegate: Unsupported Reduction Operator #%d node #%d: ",
                reduceOperatorCode, nodeIndex);
        return kTfLiteError;
    }

    bool isSupported = false;
    armnn::BackendId setBackend;
    auto validateFunc = [&](const armnn::TensorInfo& outInfo, bool& isSupported)
    {
        FORWARD_LAYER_OPAQUE_SUPPORT_FUNC("REDUCE",
                                           tfLiteContext,
                                           IsReduceSupported,
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

    // Add an Reduce layer
    armnn::IConnectableLayer* layer = delegateData.m_Network->AddReduceLayer(desc);
    layer->SetBackendId(setBackend);
    ARMNN_ASSERT(layer != nullptr);

    armnn::IOutputSlot& outputSlot = layer->GetOutputSlot(0);
    outputSlot.SetTensorInfo(outputTensorInfo);

    // try to connect the Constant Inputs if there are any
    if(ProcessInputs(layer,delegateData, tfLiteContext, tfLiteNode) != kTfLiteOk )
    {
        return kTfLiteError;
    }

    // Connect
    return Connect(layer, tfLiteContext, tfLiteNode, delegateData);
}

} // namespace armnnOpaqueDelegate
