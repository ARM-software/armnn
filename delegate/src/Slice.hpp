//
// Copyright © 2022-2023 Arm Ltd and Contributors. All rights reserved.
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
    TF_LITE_ENSURE_STATUS(ValidateNumInputs(tfLiteContext, tfLiteNode, 3, nodeIndex));
    TF_LITE_ENSURE_STATUS(ValidateNumOutputs(tfLiteContext, tfLiteNode, 1, nodeIndex));

    // Read inputs [input, begin, size]
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

    // We save the begin and size tensors in our descriptor. Therefore we have to read those values from inputs
    int inputRank = tfLiteInputs[0]->dims->size;
    auto ReadInt32Input = [&](int inputIndex, std::vector<uint32_t>& outputData) ->  TfLiteStatus
    {
        if (tfLiteInputs[inputIndex]->type != kTfLiteInt32)
        {
            TF_LITE_MAYBE_KERNEL_LOG(
                    tfLiteContext,
                    "TfLiteArmnnDelegate: The Begin- and Size-Tensors of the Slice operation need to "
                    "be of type int32. Operator: #%d node #%d: ",
                    sliceOperatorCode, nodeIndex);
            return kTfLiteError;
        }
        int rank = tfLiteInputs[inputIndex]->dims->size;
        if (rank != 1)
        {
            TF_LITE_MAYBE_KERNEL_LOG(
                    tfLiteContext,
                    "TfLiteArmnnDelegate: The Begin- and Size-Tensors of the Slice operation need to "
                    "be a 1D-Tensor. Operator: #%d node #%d: ",
                    sliceOperatorCode, nodeIndex);
            return kTfLiteError;
        }
        int numValues = tfLiteInputs[inputIndex]->dims->data[0];
        if (numValues != inputRank)
        {
            TF_LITE_MAYBE_KERNEL_LOG(
                    tfLiteContext,
                    "TfLiteArmnnDelegate: The number of values in the Begin- and Size-Tensors of the "
                    "Slice operation need to be equal to the rank of the Input-Tensor. Operator: #%d node #%d: ",
                    sliceOperatorCode, nodeIndex);
            return kTfLiteError;
        }
        // return tensor data
        auto* tensorDataPtr = tflite::GetTensorData<uint32_t>(tfLiteInputs[inputIndex]);
        outputData.assign(tensorDataPtr, tensorDataPtr+numValues);
        return kTfLiteOk;
    };

    std::vector<uint32_t> begin;
    if (ReadInt32Input(1, begin) != kTfLiteOk)
        return kTfLiteError;
    std::vector<uint32_t> size;
    if (ReadInt32Input(2, size) != kTfLiteOk)
        return kTfLiteError;

    // Write all data to the descriptor
    armnn::SliceDescriptor descriptor(begin, size);

    // Validate output
    const TfLiteTensor& tfLiteOutputTensor = tfLiteTensors[tfLiteNode->outputs->data[0]];
    if (!IsValid(tfLiteContext, tfLiteOutputTensor, sliceOperatorCode, nodeIndex))
    {
        return kTfLiteError;
    }

    const armnn::TensorInfo& inputTensorInfo  = GetTensorInfoForTfLiteTensor(*tfLiteInputs[0]);
    const armnn::TensorInfo& outputTensorInfo = GetTensorInfoForTfLiteTensor(tfLiteOutputTensor, true);

    bool isSupported = false;
    armnn::BackendId setBackend;
    auto validateFunc = [&](const armnn::TensorInfo& outInfo, bool& isSupported)
    {
        FORWARD_LAYER_SUPPORT_FUNC("SLICE",
                                   tfLiteContext,
                                   IsSliceSupported,
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

    // Add a Slice layer
    armnn::IConnectableLayer* layer = delegateData.m_Network->AddSliceLayer(descriptor);
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
    return Connect(layer, tfLiteNode, delegateData);
}

} // namespace armnnDelegate

