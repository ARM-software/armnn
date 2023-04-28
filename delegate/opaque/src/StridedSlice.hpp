//
// Copyright © 2023 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include <OpaqueDelegateUtils.hpp>

namespace armnnOpaqueDelegate
{

TfLiteStatus VisitStridedSliceOperator(DelegateData& delegateData,
                                       TfLiteOpaqueContext* tfLiteContext,
                                       TfLiteOpaqueNode* tfLiteNode,
                                       int nodeIndex,
                                       int32_t tfLiteStridedSliceOperatorCode)
{
    TF_LITE_ENSURE_STATUS(ValidateNumInputs(tfLiteContext, tfLiteNode, 4, nodeIndex));
    TF_LITE_ENSURE_STATUS(ValidateNumOutputs(tfLiteContext, tfLiteNode, 1, nodeIndex));

    // Read inputs [input, begin, end, strides]
    // Gather input indices and use to get input tensor.
    const int* inputTensors;
    int numInputs;
    if (TfLiteOpaqueNodeInputs(tfLiteNode, &inputTensors, &numInputs) != kTfLiteOk)
    {
        TF_LITE_OPAQUE_MAYBE_KERNEL_LOG(
                tfLiteContext,
                "TfLiteArmnnOpaqueDelegate: Unable to gather input tensor indices from node #%d: ",
                nodeIndex);
        return kTfLiteError;
    }

    std::vector<const TfLiteOpaqueTensor*> tfLiteInputTensors;
    tfLiteInputTensors.reserve(numInputs);
    for (int i = 0; i < numInputs; i++)
    {
        const TfLiteOpaqueTensor* inputTensor = TfLiteOpaqueContextGetOpaqueTensor(tfLiteContext, inputTensors[i]);
        tfLiteInputTensors.push_back(inputTensor);
        if (!IsValid(tfLiteContext, inputTensor, tfLiteStridedSliceOperatorCode, nodeIndex))
        {
            return kTfLiteError;
        }
    }

    const armnn::TensorInfo& inputTensorInfo  = GetTensorInfoForTfLiteOpaqueTensor(tfLiteInputTensors[0]);

    // We save the begin, end and strides tensors in our descriptor. Therefore we have to read those values from inputs
    unsigned int inputRank = inputTensorInfo.GetNumDimensions();
    auto ReadInt32Input = [&](int inputIndex, std::vector<int32_t>& outputData) ->  TfLiteStatus
    {
        if (TfLiteOpaqueTensorType(tfLiteInputTensors[inputIndex]) != kTfLiteInt32)
        {
            TF_LITE_OPAQUE_MAYBE_KERNEL_LOG(
                    tfLiteContext,
                    "TfLitearmnnOpaqueDelegate: The Begin-, End- and Stride-Tensors of the StridedSlice operation need"
                    " to be of type int32. Operator: #%d node #%d: ",
                    tfLiteStridedSliceOperatorCode, nodeIndex);
            return kTfLiteError;
        }
        uint32_t rank = TfLiteOpaqueTensorNumDims(tfLiteInputTensors[inputIndex]);
        if (rank != 1)
        {
            TF_LITE_OPAQUE_MAYBE_KERNEL_LOG(
                    tfLiteContext,
                    "TfLitearmnnOpaqueDelegate: The Begin-, End- and Stride-Tensors of the StridedSlice operation need"
                    " to be a 1D-Tensor. Operator: #%d node #%d: ",
                    tfLiteStridedSliceOperatorCode, nodeIndex);
            return kTfLiteError;
        }
        uint32_t numValues = TfLiteOpaqueTensorDim(tfLiteInputTensors[inputIndex], 0);
        if (numValues != inputRank)
        {
            TF_LITE_OPAQUE_MAYBE_KERNEL_LOG(
                   tfLiteContext,
                   "TfLitearmnnOpaqueDelegate: The number of values in the Begin-, End- and Stride-Tensors of the "
                   "StridedSlice operation need to be equal to the rank of the Input-Tensor. Operator: #%d node #%d: ",
                   tfLiteStridedSliceOperatorCode, nodeIndex);
            return kTfLiteError;
        }
        // return tensor data
        auto* tensorDataPtr = static_cast<uint32_t*>(TfLiteOpaqueTensorData(tfLiteInputTensors[inputIndex]));
        outputData.assign(tensorDataPtr, tensorDataPtr + numValues);
        return kTfLiteOk;
    };

    std::vector<int32_t> beginData;
    if (ReadInt32Input(1, beginData) != kTfLiteOk)
        return kTfLiteError;
    std::vector<int32_t> endData;
    if (ReadInt32Input(2, endData) != kTfLiteOk)
        return kTfLiteError;
    std::vector<int32_t> strideData;
    if (ReadInt32Input(3, strideData) != kTfLiteOk)
        return kTfLiteError;

    // parse built in options
    auto* nodeParameters = reinterpret_cast<TfLiteStridedSliceParams*>(TfLiteOpaqueNodeGetBuiltinData(tfLiteNode));

    // Write all data to the descriptor
    armnn::StridedSliceDescriptor descriptor;
    descriptor.m_Begin          = std::move(beginData);
    descriptor.m_End            = std::move(endData);
    descriptor.m_Stride         = std::move(strideData);
    descriptor.m_BeginMask      = nodeParameters->begin_mask;
    descriptor.m_EllipsisMask   = nodeParameters->ellipsis_mask;
    descriptor.m_EndMask        = nodeParameters->end_mask;
    descriptor.m_NewAxisMask    = nodeParameters->new_axis_mask;
    descriptor.m_ShrinkAxisMask = nodeParameters->shrink_axis_mask;
    descriptor.m_DataLayout     = armnn::DataLayout::NHWC;

    // Validate output
    // Gather output indices and use to get output tensor.
    const int* outputTensors;
    int numOutputs;
    if (TfLiteOpaqueNodeOutputs(tfLiteNode, &outputTensors, &numOutputs) != kTfLiteOk)
    {
        TF_LITE_OPAQUE_MAYBE_KERNEL_LOG(
                tfLiteContext,
                "TfLiteArmnnOpaqueDelegate: Unable to gather output tensor indices from node #%d: ",
                nodeIndex);
        return kTfLiteError;
    }

    const TfLiteOpaqueTensor* tfLiteOutputTensor = TfLiteOpaqueContextGetOpaqueTensor(tfLiteContext, outputTensors[0]);
    if (!IsValid(tfLiteContext, tfLiteOutputTensor, tfLiteStridedSliceOperatorCode, nodeIndex))
    {
        return kTfLiteError;
    }

    const armnn::TensorInfo& outputTensorInfo = GetTensorInfoForTfLiteOpaqueTensor(tfLiteOutputTensor);

    bool isSupported = false;
    armnn::BackendId setBackend;
    auto validateFunc = [&](const armnn::TensorInfo& outInfo, bool& isSupported)
    {
        FORWARD_LAYER_OPAQUE_SUPPORT_FUNC("STRIDED_SLICE",
                                          tfLiteContext,
                                          IsStridedSliceSupported,
                                          delegateData.m_Backends,
                                          isSupported,
                                          setBackend,
                                          inputTensorInfo,
                                          outInfo,
                                          descriptor);
    };

    if (!delegateData.m_Network)
    {
        validateFunc(outputTensorInfo, isSupported);
        return isSupported ? kTfLiteOk : kTfLiteError;
    }

    // Add a StridedSlice layer
    armnn::IConnectableLayer* layer = delegateData.m_Network->AddStridedSliceLayer(descriptor);
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

