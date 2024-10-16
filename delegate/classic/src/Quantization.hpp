//
// Copyright © 2022-2024 Arm Ltd and Contributors. All rights reserved.
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

TfLiteStatus VisitDequantizeOperator(DelegateData& delegateData,
                                     TfLiteContext* tfLiteContext,
                                     TfLiteNode* tfLiteNode,
                                     int nodeIndex,
                                     int32_t tfLiteDequantizeOperatorCode)
{
    TF_LITE_ENSURE_STATUS(ValidateNumInputs(tfLiteContext, tfLiteNode, 1, nodeIndex));
    TF_LITE_ENSURE_STATUS(ValidateNumOutputs(tfLiteContext, tfLiteNode, 1, nodeIndex));
    const TfLiteTensor* tfLiteTensors = tfLiteContext->tensors;
    const TfLiteTensor& tfLiteInputTensor = tfLiteTensors[tfLiteNode->inputs->data[0]];
    if (IsDynamicTensor(tfLiteInputTensor))
    {
        TF_LITE_MAYBE_KERNEL_LOG(
            tfLiteContext,
            "TfLiteArmnnDelegate: Dynamic input tensors are not supported in operator #%d node #%d: ",
            tfLiteDequantizeOperatorCode, nodeIndex);
        return kTfLiteError;
    }
    const TfLiteTensor& tfLiteOutputTensor = tfLiteTensors[tfLiteNode->outputs->data[0]];
    if (IsDynamicTensor(tfLiteOutputTensor))
    {
        TF_LITE_MAYBE_KERNEL_LOG(
            tfLiteContext,
            "TfLiteArmnnDelegate: Dynamic output tensors are not supported in operator #%d node #%d: ",
            tfLiteDequantizeOperatorCode, nodeIndex);

        return kTfLiteError;
    }
    const armnn::TensorInfo& inputTensorInfo = GetTensorInfoForTfLiteTensor(tfLiteInputTensor);
    armnn::TensorInfo outputTensorInfo = GetTensorInfoForTfLiteTensor(tfLiteOutputTensor, true);

    UpdateConstantTensorOutputs(inputTensorInfo, outputTensorInfo);

    bool isSupported = false;
    armnn::BackendId setBackend;
    auto validateFunc = [&](const armnn::TensorInfo& outputTensorInfo, bool& isSupported)
    {
        // If this is a Dequantize with a Constant input then will be replaced by a Constant layer that contains the
        // dequantized values during optimization so there's no need to check if it can be supported by the backend
        if (tflite::IsConstantTensor(&tfLiteInputTensor))
        {
            isSupported = true;
        }
        else
        {
            FORWARD_LAYER_SUPPORT_FUNC("DEQUANTIZE",
                                       tfLiteContext,
                                       IsDequantizeSupported,
                                       delegateData.m_Backends,
                                       isSupported,
                                       setBackend,
                                       inputTensorInfo,
                                       outputTensorInfo);
        }
    };

    if (!delegateData.m_Network)
    {
        validateFunc(outputTensorInfo, isSupported);
        return isSupported ? kTfLiteOk : kTfLiteError;
    }

    auto layerName = GetLayerName(armnn::LayerType::Dequantize, nodeIndex);
    armnn::IConnectableLayer* dequantizeLayer = delegateData.m_Network->AddDequantizeLayer(layerName.c_str());
    dequantizeLayer->SetBackendId(setBackend);
    ARMNN_ASSERT(dequantizeLayer != nullptr);

    armnn::IOutputSlot& outputSlot = dequantizeLayer->GetOutputSlot(0);
    outputSlot.SetTensorInfo(outputTensorInfo);

    auto inputsTensorsProcess = ProcessInputs(dequantizeLayer,
                                              delegateData,
                                              tfLiteContext,
                                              tfLiteNode,
                                              nodeIndex);
    if (inputsTensorsProcess == kTfLiteError)
    {
        return inputsTensorsProcess;
    }

    return Connect(dequantizeLayer, tfLiteNode, delegateData);
}

TfLiteStatus VisitQuantizeOperator(DelegateData& delegateData,
                                   TfLiteContext* tfLiteContext,
                                   TfLiteNode* tfLiteNode,
                                   int nodeIndex,
                                   int32_t tfLiteQuantizeOperatorCode)
{
    TF_LITE_ENSURE_STATUS(ValidateNumInputs(tfLiteContext, tfLiteNode, 1, nodeIndex));
    TF_LITE_ENSURE_STATUS(ValidateNumOutputs(tfLiteContext, tfLiteNode, 1, nodeIndex));

    const TfLiteTensor* tfLiteTensors = tfLiteContext->tensors;
    const TfLiteTensor& tfLiteInputTensor = tfLiteTensors[tfLiteNode->inputs->data[0]];
    if (IsDynamicTensor(tfLiteInputTensor))
    {
        TF_LITE_MAYBE_KERNEL_LOG(
            tfLiteContext,
            "TfLiteArmnnDelegate: Dynamic input tensors are not supported in operator #%d node #%d: ",
            tfLiteQuantizeOperatorCode, nodeIndex);
        return kTfLiteError;
    }

    const TfLiteTensor& tfLiteOutputTensor = tfLiteTensors[tfLiteNode->outputs->data[0]];
    if (IsDynamicTensor(tfLiteOutputTensor))
    {
        TF_LITE_MAYBE_KERNEL_LOG(
            tfLiteContext,
            "TfLiteArmnnDelegate: Dynamic output tensors are not supported in operator #%d node #%d: ",
            tfLiteQuantizeOperatorCode, nodeIndex);
        return kTfLiteError;
    }

    // Only affine per-layer quantization is supported.
    if (!IsAffineQuantization(tfLiteOutputTensor))
    {
        TF_LITE_MAYBE_KERNEL_LOG(
            tfLiteContext,
            "TfLiteArmnnDelegate: Only affine per-layer quantization is supported in operator #%d node #%d: ",
            tfLiteQuantizeOperatorCode, nodeIndex);
        return kTfLiteError;
    }

    const armnn::TensorInfo& inputTensorInfo = GetTensorInfoForTfLiteTensor(tfLiteInputTensor);
    const armnn::TensorInfo& outputTensorInfo = GetTensorInfoForTfLiteTensor(tfLiteOutputTensor, true);

    if (outputTensorInfo.HasPerAxisQuantization())
    {
        auto outputTensorDataType = outputTensorInfo.GetDataType();
        if (outputTensorDataType == armnn::DataType::QAsymmU8 || outputTensorDataType == armnn::DataType::QAsymmS8)
        {
            TF_LITE_MAYBE_KERNEL_LOG(
                tfLiteContext,
                "TfLiteArmnnDelegate: Per Axis Quantization is not supported in asymmetric Quantization Datatype.");
            return kTfLiteError;
        }
    }

    bool isSupported = false;
    armnn::BackendId setBackend;
    auto validateFunc = [&](const armnn::TensorInfo& outputTensorInfo, bool& isSupported)
    {
        FORWARD_LAYER_SUPPORT_FUNC("QUANTIZE",
                                   tfLiteContext,
                                   IsQuantizeSupported,
                                   delegateData.m_Backends,
                                   isSupported,
                                   setBackend,
                                   inputTensorInfo,
                                   outputTensorInfo);
    };

    if (!delegateData.m_Network)
    {
        validateFunc(outputTensorInfo, isSupported);
        return isSupported ? kTfLiteOk : kTfLiteError;
    }

    auto layerName = GetLayerName(armnn::LayerType::Quantize, nodeIndex);
    armnn::IConnectableLayer* quantizeLayer = delegateData.m_Network->AddQuantizeLayer(layerName.c_str());
    quantizeLayer->SetBackendId(setBackend);
    ARMNN_ASSERT(quantizeLayer != nullptr);

    armnn::IOutputSlot& outputSlot = quantizeLayer->GetOutputSlot(0);
    outputSlot.SetTensorInfo(outputTensorInfo);

    // try to connect the Constant Inputs if there are any
    if (ProcessInputs(quantizeLayer, delegateData, tfLiteContext, tfLiteNode, nodeIndex) != kTfLiteOk)
    {
        return kTfLiteError;
    }

    return Connect(quantizeLayer, tfLiteNode, delegateData);
}

} // namespace armnnDelegate
