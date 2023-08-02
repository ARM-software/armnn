//
// Copyright © 2023 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include <armnn/utility/IgnoreUnused.hpp>

#include <tensorflow/lite/builtin_ops.h>
#include <tensorflow/lite/c/builtin_op_data.h>
#include <tensorflow/lite/c/common.h>
#include <tensorflow/lite/minimal_logging.h>
#include <tensorflow/lite/kernels/internal/tensor_ctypes.h>
#include <tensorflow/lite/schema/schema_generated.h>

namespace armnnDelegate
{
TfLiteStatus ValidateTileOperator(DelegateData& delegateData,
                                  TfLiteContext* tfLiteContext,
                                  const armnn::TensorInfo& inputInfo,
                                  const armnn::TensorInfo& outputInfo,
                                  const armnn::TileDescriptor& descriptor)
{
    bool isSupported = false;
    FORWARD_LAYER_SUPPORT_FUNC("TILE",
                               tfLiteContext,
                               IsTileSupported,
                               delegateData.m_Backends,
                               isSupported,
                               armnn::BackendId(),
                               inputInfo,
                               outputInfo,
                               descriptor);
    return isSupported ? kTfLiteOk : kTfLiteError;
}

TfLiteStatus VisitTileOperator(DelegateData& delegateData,
                               TfLiteContext* tfLiteContext,
                               TfLiteNode* tfLiteNode,
                               int nodeIndex,
                               int32_t tileOperatorCode)
{
    TF_LITE_ENSURE_STATUS(ValidateNumInputs(tfLiteContext, tfLiteNode, 2, nodeIndex));
    TF_LITE_ENSURE_STATUS(ValidateNumOutputs(tfLiteContext, tfLiteNode, 1, nodeIndex));

    const TfLiteTensor* tfLiteTensors = tfLiteContext->tensors;

    // The input contains the data that should be tiled
    const TfLiteTensor& tfLiteInputTensor = tfLiteTensors[tfLiteNode->inputs->data[0]];
    if (IsDynamicTensor(tfLiteInputTensor))
    {
        TF_LITE_MAYBE_KERNEL_LOG(
            tfLiteContext,
            "TfLiteArmnnDelegate: Dynamic input tensors are not supported in operator #%d node #%d: ",
            tileOperatorCode, nodeIndex);
        return kTfLiteError;
    }

    // The multiples tensor contains the number of copies for each axis
    const TfLiteTensor& tfLiteMultiplesTensor = tfLiteTensors[tfLiteNode->inputs->data[1]];
    if (IsDynamicTensor(tfLiteMultiplesTensor))
    {
        TF_LITE_MAYBE_KERNEL_LOG(
            tfLiteContext,
            "TfLiteArmnnDelegate: Dynamic input tensors are not supported in operator #%d node #%d: ",
            tileOperatorCode, nodeIndex);
        return kTfLiteError;
    }

    // The output tensor
    const TfLiteTensor& tfLiteOutputTensor = tfLiteTensors[tfLiteNode->outputs->data[0]];
    if (IsDynamicTensor(tfLiteOutputTensor))
    {
        TF_LITE_MAYBE_KERNEL_LOG(
            tfLiteContext,
            "TfLiteArmnnDelegate: Dynamic output tensors are not supported in operator #%d node #%d: ",
            tileOperatorCode, nodeIndex);
        return kTfLiteError;
    }

    const armnn::TensorInfo& inputTensorInfo = GetTensorInfoForTfLiteTensor(tfLiteInputTensor);
    const armnn::TensorInfo& multiplesTensorInfo = GetTensorInfoForTfLiteTensor(tfLiteMultiplesTensor);
    const armnn::TensorInfo& outputTensorInfo = GetTensorInfoForTfLiteTensor(tfLiteOutputTensor);

    // Multiples length must be the same as the number of dimension in input tensor
    if (multiplesTensorInfo.GetNumElements() != inputTensorInfo.GetNumDimensions())
    {
        TF_LITE_MAYBE_KERNEL_LOG(
            tfLiteContext,
            "TfLiteArmnnDelegate: The Multiples length must be the same as the number of dimension in input tensor",
            "Operator: #%d node #%d: ",
            tileOperatorCode, nodeIndex);
        return kTfLiteError;
    }

    // Get the Multiples data: In armnn, the values of the multiples input tensor is saved in the operator descriptor
    // We have to read it from the input tensor and write it the descriptor
    auto* multiplesTensorDataPtr = tflite::GetTensorData<int32_t>(&tfLiteMultiplesTensor);
    auto multiplesTensorNum = tfLiteMultiplesTensor.dims->data[0];
    std::vector<int32_t> multiplesIntData(multiplesTensorDataPtr, multiplesTensorDataPtr + multiplesTensorNum);

    // The multiples must be positive
    for (auto multiple : multiplesIntData)
    {
        if (multiple < 0)
        {
            TF_LITE_MAYBE_KERNEL_LOG(
                tfLiteContext,
                "TfLiteArmnnDelegate: The Multiples must be positive values",
                "Operator: #%d node #%d: ",
                tileOperatorCode, nodeIndex);
            return kTfLiteError;
        }
    }

    // The original input from TFLite is int32, and we have to make it as uint32 for our descriptor
    std::vector<uint32_t> multiplesUintData;
    std::transform(multiplesIntData.begin(),
                   multiplesIntData.end(),
                   std::back_inserter(multiplesUintData),
                   [] (const int value)
                   {
                        return static_cast<uint32_t>(value);
                   });

    armnn::TileDescriptor tileDescriptor;
    tileDescriptor.m_Multiples = multiplesUintData;

    // Check output dimensions
    if (inputTensorInfo.GetNumDimensions() != outputTensorInfo.GetNumDimensions())
    {
        TF_LITE_MAYBE_KERNEL_LOG(
            tfLiteContext,
            "TfLiteArmnnDelegate: Input tensor dimension and output tensor dimension differ",
            "Operator: #%d node #%d: ",
            tileOperatorCode, nodeIndex);
        return kTfLiteError;
    }

    // No network pointer indicates that only support for this operator should be checked
    if (!delegateData.m_Network)
    {
        return ValidateTileOperator(delegateData,
                                    tfLiteContext,
                                    inputTensorInfo,
                                    outputTensorInfo,
                                    tileDescriptor);
    }

    auto layerName = GetLayerName(armnn::LayerType::Tile, nodeIndex);
    armnn::IConnectableLayer* layer = delegateData.m_Network->AddTileLayer(tileDescriptor, layerName.c_str());

    if (layer == nullptr)
    {
        return kTfLiteError;
    }

    layer->GetOutputSlot(0).SetTensorInfo(outputTensorInfo);

    if (ProcessInputs(layer, delegateData, tfLiteContext, tfLiteNode, nodeIndex) != kTfLiteOk)
    {
        return kTfLiteError;
    }

    return Connect(layer, tfLiteNode, delegateData);
}

} // namespace armnnDelegate