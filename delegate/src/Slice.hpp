//
// Copyright © 2020 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include <armnn/utility/IgnoreUnused.hpp>

#include <tensorflow/lite/builtin_ops.h>
#include <tensorflow/lite/c/builtin_op_data.h>
#include <tensorflow/lite/c/common.h>
#include <tensorflow/lite/minimal_logging.h>

namespace armnnDelegate
{

TfLiteStatus VisitSliceOperator(DelegateData& delegateData,
                                TfLiteContext* tfLiteContext,
                                TfLiteNode* tfLiteNode,
                                int nodeIndex,
                                int32_t sliceOperatorCode)
{
    TF_LITE_ENSURE_STATUS(ValidateNumInputs(tfLiteContext, tfLiteNode, 4, nodeIndex));
    TF_LITE_ENSURE_STATUS(ValidateNumOutputs(tfLiteContext, tfLiteNode, 1, nodeIndex));

    // Read inputs [input, begin, end, strides]
    int numInputs = tfLiteNode->inputs->size;
    std::vector<const TfLiteTensor*> tfLiteInputs;
    tfLiteInputs.reserve(numInputs);
    const TfLiteTensor* tfLiteTensors = tfLiteContext->tensors;
    for (int i = 0; i < numInputs; i++)
    {
        const TfLiteTensor* inputTensor = &tfLiteTensors[tfLiteNode->inputs->data[i]];
        tfLiteInputs.push_back(inputTensor);
        if (!IsValid(tfLiteContext, *inputTensor, sliceOperatorCode, nodeIndex))
        {
            return kTfLiteError;
        }
    }

    // We save the begin, end and strides tensors in our descriptor. Therefore we have to read those values from inputs
    int inputRank = tfLiteInputs[0]->dims->size;
    auto ReadInt32Input = [&](int inputIndex, std::vector<int32_t>& outputData) ->  TfLiteStatus
    {
        if (tfLiteInputs[inputIndex]->type != kTfLiteInt32)
        {
            TF_LITE_MAYBE_KERNEL_LOG(
                    tfLiteContext,
                    "TfLiteArmnnDelegate: The Begin-, End- and Stride-Tensors of the StridedSlice operation need to "
                    "be of type int32. Operator: #%d node #%d: ",
                    sliceOperatorCode, nodeIndex);
            return kTfLiteError;
        }
        int rank = tfLiteInputs[inputIndex]->dims->size;
        if (rank != 1)
        {
            TF_LITE_MAYBE_KERNEL_LOG(
                    tfLiteContext,
                    "TfLiteArmnnDelegate: The Begin-, End- and Stride-Tensors of the StridedSlice operation need to "
                    "be a 1D-Tensor. Operator: #%d node #%d: ",
                    sliceOperatorCode, nodeIndex);
            return kTfLiteError;
        }
        int numValues = tfLiteInputs[inputIndex]->dims->data[0];
        if (numValues != inputRank)
        {
            TF_LITE_MAYBE_KERNEL_LOG(
                    tfLiteContext,
                    "TfLiteArmnnDelegate: The number of values in the Begin-, End- and Stride-Tensors of the "
                    "StridedSlice operation need to be equal to the rank of the Input-Tensor. Operator: #%d node #%d: ",
                    sliceOperatorCode, nodeIndex);
            return kTfLiteError;
        }
        // return tensor data
        auto* tensorDataPtr = tflite::GetTensorData<int32_t>(tfLiteInputs[inputIndex]);
        outputData.assign(tensorDataPtr, tensorDataPtr+numValues);
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
    auto* stridedSliceParams = reinterpret_cast<TfLiteStridedSliceParams*>(tfLiteNode->builtin_data);

    // Write all data to the descriptor
    armnn::StridedSliceDescriptor descriptor;
    descriptor.m_Begin          = std::move(beginData);
    descriptor.m_End            = std::move(endData);
    descriptor.m_Stride         = std::move(strideData);
    descriptor.m_BeginMask      = stridedSliceParams->begin_mask;
    descriptor.m_EllipsisMask   = stridedSliceParams->ellipsis_mask;
    descriptor.m_EndMask        = stridedSliceParams->end_mask;
    descriptor.m_NewAxisMask    = stridedSliceParams->new_axis_mask;
    descriptor.m_ShrinkAxisMask = stridedSliceParams->shrink_axis_mask;
    descriptor.m_DataLayout     = armnn::DataLayout::NHWC;

    // Validate output
    const TfLiteTensor& tfLiteOutputTensor = tfLiteTensors[tfLiteNode->outputs->data[0]];
    if (!IsValid(tfLiteContext, tfLiteOutputTensor, sliceOperatorCode, nodeIndex))
    {
        return kTfLiteError;
    }

    const armnn::TensorInfo& inputTensorInfo  = GetTensorInfoForTfLiteTensor(*tfLiteInputs[0]);
    const armnn::TensorInfo& outputTensorInfo = GetTensorInfoForTfLiteTensor(tfLiteOutputTensor);

    bool isSupported = false;
    auto validateFunc = [&](const armnn::TensorInfo& outInfo, bool& isSupported)
    {
        FORWARD_LAYER_SUPPORT_FUNC(__func__,
                                   tfLiteContext,
                                   IsStridedSliceSupported,
                                   delegateData.m_Backends,
                                   isSupported,
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
    ARMNN_ASSERT(layer != nullptr);

    armnn::IOutputSlot& outputSlot = layer->GetOutputSlot(0);
    outputSlot.SetTensorInfo(outputTensorInfo);

    // Connect
    return Connect(layer, tfLiteNode, delegateData);
}

} // namespace armnnDelegate
