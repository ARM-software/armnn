//
// Copyright © 2022-2023 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include <armnn/utility/IgnoreUnused.hpp>

#include <ClassicDelegateUtils.hpp>

#include <tensorflow/lite/builtin_ops.h>
#include <tensorflow/lite/c/builtin_op_data.h>
#include <tensorflow/lite/c/common.h>
#include <tensorflow/lite/minimal_logging.h>
#include <numeric>

namespace armnnDelegate
{

TfLiteStatus VisitCastOperator(DelegateData& delegateData,
                               TfLiteContext* tfLiteContext,
                               TfLiteNode* tfLiteNode,
                               int nodeIndex,
                               int32_t operatorCode)
{
    TF_LITE_ENSURE_STATUS(ValidateNumInputs(tfLiteContext, tfLiteNode, 1, nodeIndex));
    TF_LITE_ENSURE_STATUS(ValidateNumOutputs(tfLiteContext, tfLiteNode, 1, nodeIndex));

    const TfLiteTensor* tfLiteTensors = tfLiteContext->tensors;
    const TfLiteTensor& tfLiteInputTensor = tfLiteTensors[tfLiteNode->inputs->data[0]];
    if (!IsValid(tfLiteContext, tfLiteInputTensor, operatorCode, nodeIndex))
    {
        return kTfLiteError;
    }

    const TfLiteTensor& tfLiteOutputTensor = tfLiteTensors[tfLiteNode->outputs->data[0]];
    if (!IsValid(tfLiteContext, tfLiteOutputTensor, operatorCode, nodeIndex))
    {
        return kTfLiteError;
    }

    const armnn::TensorInfo& inputTensorInfo  = GetTensorInfoForTfLiteTensor(tfLiteInputTensor);
    const armnn::TensorInfo& outputTensorInfo = GetTensorInfoForTfLiteTensor(tfLiteOutputTensor, true);

    bool isSupported = false;
    armnn::BackendId setBackend;
    auto validateFunc = [&](const armnn::TensorInfo& outInfo, bool& isSupported)
    {
        FORWARD_LAYER_SUPPORT_FUNC("CAST",
                                   tfLiteContext,
                                   IsCastSupported,
                                   delegateData.m_Backends,
                                   isSupported,
                                   setBackend,
                                   inputTensorInfo,
                                   outInfo);
    };

    // If the m_Network is a nullptr, this signals that a prerequisite TfLite callback is required to clarify the
    // support for the operator
    // If supported, VisitCastOperator will be called again to add the layer to the network as seen further below
    if (!delegateData.m_Network)
    {
        validateFunc(outputTensorInfo, isSupported);
        return isSupported ? kTfLiteOk : kTfLiteError;
    }

    // Add a Cast layer
    armnn::IConnectableLayer* layer = delegateData.m_Network->AddCastLayer();
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


TfLiteStatus CreateOutputTensorShape(const armnn::TensorInfo& inputTensorInfo,
                                     const std::vector<int32_t>& targetShape,
                                     armnn::ReshapeDescriptor& reshapeDesc)
{
    std::vector<unsigned int> outputDims(targetShape.begin(), targetShape.end());
    const auto stretchDim = std::find(targetShape.begin(), targetShape.end(), -1);

    if (stretchDim != targetShape.end())
    {
        if (std::find(std::next(stretchDim), targetShape.end(), -1) != targetShape.end())
        {
            // Return kTfLiteError and log the error after returning
            return kTfLiteError;
        }

        auto targetNumElements =
            armnn::numeric_cast<unsigned int>(
                std::accumulate(targetShape.begin(), targetShape.end(), -1, std::multiplies<int32_t>()));

        auto stretchIndex = static_cast<size_t>(std::distance(targetShape.begin(), stretchDim));
        outputDims[stretchIndex] = inputTensorInfo.GetNumElements() / targetNumElements;
    }

    armnn::TensorShape outputShape = armnn::TensorShape(static_cast<unsigned int>(outputDims.size()),
                                                        outputDims.data());
    reshapeDesc.m_TargetShape = outputShape;
    return kTfLiteOk;
}

TfLiteStatus VisitReshapeOperator(DelegateData& delegateData,
                                  TfLiteContext* tfLiteContext,
                                  TfLiteNode* tfLiteNode,
                                  int nodeIndex,
                                  int32_t operatorCode)
{
    auto numInputs = tfLiteNode->inputs->size;

    if (numInputs == 2)
    {
        TF_LITE_ENSURE_STATUS(ValidateNumInputs(tfLiteContext, tfLiteNode, 2, nodeIndex));
    }
    else
    {
        TF_LITE_ENSURE_STATUS(ValidateNumInputs(tfLiteContext, tfLiteNode, 1, nodeIndex));
    }
    TF_LITE_ENSURE_STATUS(ValidateNumOutputs(tfLiteContext, tfLiteNode, 1, nodeIndex));

    const TfLiteTensor* tfLiteTensors = tfLiteContext->tensors;
    const TfLiteTensor& tfLiteInputTensor0 = tfLiteTensors[tfLiteNode->inputs->data[0]];
    if (!IsValid(tfLiteContext, tfLiteInputTensor0, operatorCode, nodeIndex))
    {
        return kTfLiteError;
    }

    const TfLiteTensor& tfLiteOutputTensor = tfLiteTensors[tfLiteNode->outputs->data[0]];
    if (!IsValid(tfLiteContext, tfLiteOutputTensor, operatorCode, nodeIndex))
    {
        return kTfLiteError;
    }

    const armnn::TensorInfo& inputTensorInfo0 = GetTensorInfoForTfLiteTensor(tfLiteInputTensor0);
    const armnn::TensorInfo& outputTensorInfo = GetTensorInfoForTfLiteTensor(tfLiteOutputTensor, true);

    armnn::ReshapeDescriptor reshapeDesc;
    std::vector<int32_t> targetShape;

    TfLiteReshapeParams* reshapeOptions = reinterpret_cast<TfLiteReshapeParams*>(tfLiteNode->builtin_data);

    // The new shape can be defined by either a second input tensor or by a builtin option, we need to check for both.
    // Options might be set without valid data. we need to check the dimensions are in a valid range.
    if (reshapeOptions && reshapeOptions->num_dimensions > 0 && reshapeOptions->num_dimensions <= 8)
    {
        for (int i=0; i < reshapeOptions->num_dimensions; ++i)
        {
            targetShape.push_back(reshapeOptions->shape[i]);
        }
    }
    else if (numInputs == 2)
    {
        // Get shape from the second input tensor
        const TfLiteTensor& tfLiteShapeInputTensor = tfLiteTensors[tfLiteNode->inputs->data[1]];
        if (!IsValid(tfLiteContext, tfLiteShapeInputTensor, operatorCode, nodeIndex))
        {
            return kTfLiteError;
        }

        if (tfLiteShapeInputTensor.dims->size != 1)
        {
            TF_LITE_MAYBE_KERNEL_LOG(tfLiteContext,
                                     "TfLiteArmnnDelegate: Target 'shape' input is not a 1D tensor in "
                                     "operator #%d node #%d: Falling back to TfLiteOptions.",
                                     operatorCode, nodeIndex);
        }
        else
        {
            // Get the shape data out of the input tensor
            auto* shapeTensorDataPtr = tflite::GetTensorData<int32_t>(&tfLiteShapeInputTensor);
            auto shapeTensorNumValues = tfLiteShapeInputTensor.dims->data[0];
            for (auto i=0; i < shapeTensorNumValues; ++i)
            {
                targetShape.push_back(*(shapeTensorDataPtr+i));
            }
        }
    }
    else
    {
        TF_LITE_MAYBE_KERNEL_LOG(tfLiteContext,
                                 "Target shape not defined in reshape parameters or input tensor. "
                                 "At least one method required in operator #%d node #%d: ",
                                 operatorCode, nodeIndex);
        return kTfLiteError;
    }

    // Use the data to create the required tensor shape.
    if (CreateOutputTensorShape(inputTensorInfo0, targetShape, reshapeDesc) != kTfLiteOk)
    {
        TF_LITE_MAYBE_KERNEL_LOG(tfLiteContext,
                                 "TfLiteArmnnDelegate: At most one component of shape can be -1 in: "
                                 "operator #%d node #%d: ",
                                 operatorCode, nodeIndex);
        return kTfLiteError;
    }

    if (reshapeDesc.m_TargetShape.GetNumElements() != inputTensorInfo0.GetNumElements())
    {
        TF_LITE_MAYBE_KERNEL_LOG(
            tfLiteContext,
            "TfLiteArmnnDelegate: Reshape, number of elements in output shape does not match input "
            "operator #%d node #%d: ",
            operatorCode, nodeIndex);
        return kTfLiteError;
    }

    bool isSupported = false;
    armnn::BackendId setBackend;
    auto validateFunc = [&](const armnn::TensorInfo& outInfo, bool& isSupported)
    {
        FORWARD_LAYER_SUPPORT_FUNC("RESHAPE",
                                   tfLiteContext,
                                   IsReshapeSupported,
                                   delegateData.m_Backends,
                                   isSupported,
                                   setBackend,
                                   inputTensorInfo0,
                                   outInfo,
                                   reshapeDesc);
    };

    if (!delegateData.m_Network)
    {
        validateFunc(outputTensorInfo, isSupported);
        return isSupported ? kTfLiteOk : kTfLiteError;
    }

    armnn::IConnectableLayer* layer = delegateData.m_Network->AddReshapeLayer(reshapeDesc);
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

TfLiteStatus VisitSqueezeOperator(DelegateData& delegateData,
                                  TfLiteContext* tfLiteContext,
                                  TfLiteNode* tfLiteNode,
                                  int nodeIndex,
                                  int32_t operatorCode)
{
    armnn::IgnoreUnused(delegateData,
                        tfLiteContext,
                        tfLiteNode,
                        nodeIndex,
                        operatorCode);

    return kTfLiteError;
}

TfLiteStatus VisitExpandDimsOperator(DelegateData& delegateData,
                                     TfLiteContext* tfLiteContext,
                                     TfLiteNode* tfLiteNode,
                                     int nodeIndex,
                                     int32_t operatorCode)
{
    armnn::IgnoreUnused(delegateData,
                        tfLiteContext,
                        tfLiteNode,
                        nodeIndex,
                        operatorCode);

    return kTfLiteError;
}

} // namespace armnnDelegate
