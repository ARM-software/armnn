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
#include <fmt/format.h>

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
    auto ReadInt32Input = [&](int inputIndex, std::vector<int32_t>& outputData, const char* name) -> TfLiteStatus
    {
        if (tfLiteInputs[inputIndex]->type != kTfLiteInt32)
        {
            TF_LITE_MAYBE_KERNEL_LOG(
                    tfLiteContext,
                    "TfLiteArmnnDelegate: The %s Tensor of the Slice operation needs to "
                    "be of type int32. Operator: #%d node #%d: ",
                    name, sliceOperatorCode, nodeIndex);
            return kTfLiteError;
        }
        int rank = tfLiteInputs[inputIndex]->dims->size;
        if (rank != 1)
        {
            TF_LITE_MAYBE_KERNEL_LOG(
                    tfLiteContext,
                    "TfLiteArmnnDelegate: The %s Tensor of the Slice operation needs to "
                    "be a 1D-Tensor. Operator: #%d node #%d: ",
                    name, sliceOperatorCode, nodeIndex);
            return kTfLiteError;
        }
        int numValues = tfLiteInputs[inputIndex]->dims->data[0];
        if (numValues != inputRank)
        {
            TF_LITE_MAYBE_KERNEL_LOG(
                    tfLiteContext,
                    "TfLiteArmnnDelegate: The number of values in the %s Tensor of the "
                    "Slice operation needs to be equal to the rank of the Input Tensor. Operator: #%d node #%d: ",
                    name, sliceOperatorCode, nodeIndex);
            return kTfLiteError;
        }
        // return tensor data
        auto* tensorDataPtr = tflite::GetTensorData<int32_t>(tfLiteInputs[inputIndex]);
        outputData.assign(tensorDataPtr, tensorDataPtr + numValues);
        return kTfLiteOk;
    };

    std::vector<int32_t> signedBegin;
    if (ReadInt32Input(1, signedBegin, "Begin") != kTfLiteOk)
    {
        return kTfLiteError;
    }

    std::vector<int32_t> signedSize;
    if (ReadInt32Input(2, signedSize, "Size") != kTfLiteOk)
    {
        return kTfLiteError;
    }
    std::vector<uint32_t> begin({ signedBegin.begin(), signedBegin.end() });
    std::vector<uint32_t> size(signedSize.size());

    for (unsigned int i = 0; i < signedSize.size(); ++i)
    {
        int signedValue = signedSize[i];
        if (signedValue < -1 || signedValue > tfLiteInputs[0]->dims->data[i] - signedBegin[i])
        {
            TF_LITE_MAYBE_KERNEL_LOG(
                    tfLiteContext,
                    "TfLiteArmnnDelegate: Invalid value for Size. Size must be in range [-1, inputDimSize - begin] "
                    "[-1, %d] inclusive but was %d Operator: #%d node #%d: ",
                    tfLiteInputs[0]->dims->data[i] - signedBegin[i], signedValue, sliceOperatorCode,
                    nodeIndex);
            return kTfLiteError;
        }
        if (signedValue == -1)
        {
            size[i] = tfLiteInputs[0]->dims->data[i] - signedBegin[i];
        }
        else
        {
            size[i] = static_cast<uint32_t>(signedValue);
        }
    }

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
    auto layerName = GetLayerName(armnn::LayerType::Slice, nodeIndex);
    armnn::IConnectableLayer* layer = delegateData.m_Network->AddSliceLayer(descriptor, layerName.c_str());
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
    return Connect(layer, tfLiteNode, delegateData);
}

} // namespace armnnDelegate

