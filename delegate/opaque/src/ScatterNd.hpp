//
// Copyright © 2024 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include <OpaqueDelegateUtils.hpp>

namespace armnnOpaqueDelegate
{
TfLiteStatus ValidateScatterNdOperator(DelegateData& delegateData,
                                       TfLiteOpaqueContext *tfLiteContext,
                                       const armnn::TensorInfo& indicesInfo,
                                       const armnn::TensorInfo& updatesInfo,
                                       const armnn::TensorInfo& shapeInfo,
                                       const armnn::TensorInfo& outputInfo,
                                       const armnn::ScatterNdDescriptor& descriptor)
{
    bool isSupported = false;
    FORWARD_LAYER_OPAQUE_SUPPORT_FUNC("SCATTER_ND",
                                      tfLiteContext,
                                      IsScatterNdSupported,
                                      delegateData.m_Backends,
                                      isSupported,
                                      armnn::BackendId(),
                                      shapeInfo,
                                      indicesInfo,
                                      updatesInfo,
                                      outputInfo,
                                      descriptor);
    return isSupported ? kTfLiteOk : kTfLiteError;
}

TfLiteStatus VisitScatterNdOperator(DelegateData& delegateData,
                                    TfLiteOpaqueContext* tfLiteContext,
                                    TfLiteOpaqueNode* tfLiteNode,
                                    int nodeIndex,
                                    int32_t scatterNdOperatorCode)
{
    TF_LITE_ENSURE_STATUS(ValidateNumInputs(tfLiteContext, tfLiteNode, 3, nodeIndex));
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

    // Gather input indices and use to get output tensor.
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

    // The indices tensor are the positions the data is updated/scattered into
    const TfLiteOpaqueTensor* tfLiteIndicesTensor = TfLiteOpaqueContextGetOpaqueTensor(tfLiteContext, inputTensors[0]);
    if (IsDynamicTensor(tfLiteIndicesTensor))
    {
        TF_LITE_OPAQUE_MAYBE_KERNEL_LOG(
            tfLiteContext,
            "TfLiteArmnnOpaqueDelegate: Dynamic input tensors are not supported in operator #%d node #%d: ",
            scatterNdOperatorCode, nodeIndex);
        return kTfLiteError;
    }

    // The updates tensor provides the data which will be updated/scattered into the relevant indices
    const TfLiteOpaqueTensor* tfLiteUpdatesTensor = TfLiteOpaqueContextGetOpaqueTensor(tfLiteContext, inputTensors[1]);
    if (IsDynamicTensor(tfLiteUpdatesTensor))
    {
        TF_LITE_OPAQUE_MAYBE_KERNEL_LOG(
            tfLiteContext,
            "TfLiteArmnnOpaqueDelegate: Dynamic input tensors are not supported in operator #%d node #%d: ",
            scatterNdOperatorCode, nodeIndex);
        return kTfLiteError;
    }

    // For TFLite ScatterNd there is no input tensor
    // The shape tensor is a 1D tensor which represents the shape of an input tensor to be filled with zeros
    const TfLiteOpaqueTensor* tfLiteShapeTensor = TfLiteOpaqueContextGetOpaqueTensor(tfLiteContext, inputTensors[2]);
    if (IsDynamicTensor(tfLiteShapeTensor))
    {
        TF_LITE_OPAQUE_MAYBE_KERNEL_LOG(
                tfLiteContext,
                "TfLiteArmnnOpaqueDelegate: Dynamic input tensors are not supported in operator #%d node #%d: ",
                scatterNdOperatorCode, nodeIndex);
        return kTfLiteError;
    }

    // The output tensor
    const TfLiteOpaqueTensor* tfLiteOutputTensor = TfLiteOpaqueContextGetOpaqueTensor(tfLiteContext, outputTensors[0]);
    if (IsDynamicTensor(tfLiteOutputTensor))
    {
        TF_LITE_OPAQUE_MAYBE_KERNEL_LOG(
            tfLiteContext,
            "TfLiteArmnnOpaqueDelegate: Dynamic output tensors are not supported in operator #%d node #%d: ",
            scatterNdOperatorCode, nodeIndex);
        return kTfLiteError;
    }

    const armnn::TensorInfo& shapeTensorInfo = GetTensorInfoForTfLiteOpaqueTensor(tfLiteShapeTensor);
    const armnn::TensorInfo& indicesTensorInfo = GetTensorInfoForTfLiteOpaqueTensor(tfLiteIndicesTensor);
    const armnn::TensorInfo& updatesTensorInfo = GetTensorInfoForTfLiteOpaqueTensor(tfLiteUpdatesTensor);
    const armnn::TensorInfo& outputTensorInfo = GetTensorInfoForTfLiteOpaqueTensor(tfLiteOutputTensor, true);

    armnn::ScatterNdDescriptor scatterNdDescriptor;
    scatterNdDescriptor.m_Function     = armnn::ScatterNdFunction::Update;
    scatterNdDescriptor.m_InputEnabled = false;
    scatterNdDescriptor.m_Axis         = 0;
    scatterNdDescriptor.m_AxisEnabled  = false;

    // Check output dimensions
    if (shapeTensorInfo.GetShape().GetNumElements() != outputTensorInfo.GetNumDimensions())
    {
        TF_LITE_OPAQUE_MAYBE_KERNEL_LOG(
                tfLiteContext,
                "TfLiteArmnnOpaqueDelegate: Input tensor dimension and output tensor dimension differ",
                "Operator: #%d node #%d: ",
                scatterNdOperatorCode, nodeIndex);
        return kTfLiteError;
    }

    // No network pointer indicates that only support for this operator should be checked
    if (!delegateData.m_Network)
    {
        return ValidateScatterNdOperator(delegateData,
                                         tfLiteContext,
                                         indicesTensorInfo,
                                         updatesTensorInfo,
                                         shapeTensorInfo,
                                         outputTensorInfo,
                                         scatterNdDescriptor);
    }

    auto layerName = GetName(armnn::LayerType::ScatterNd, nodeIndex);
    armnn::IConnectableLayer* layer = delegateData.m_Network->AddScatterNdLayer(scatterNdDescriptor, layerName.c_str());

    if (layer == nullptr)
    {
        return kTfLiteError;
    }

    layer->GetOutputSlot(0).SetTensorInfo(outputTensorInfo);

    if (ProcessInputs(layer, delegateData, tfLiteContext, tfLiteNode, nodeIndex) != kTfLiteOk)
    {
        return kTfLiteError;
    }

    delegateData.m_OutputSlotForNode[inputTensors[2]]->Connect(layer->GetInputSlot(0));
    delegateData.m_OutputSlotForNode[inputTensors[0]]->Connect(layer->GetInputSlot(1));
    delegateData.m_OutputSlotForNode[inputTensors[1]]->Connect(layer->GetInputSlot(2));

    // Prepare output slots
    armnn::IOutputSlot& outputSlot = layer->GetOutputSlot(0);
    delegateData.m_OutputSlotForNode[static_cast<unsigned long>(outputTensors[0])] = &outputSlot;

    return kTfLiteOk;
}

} // namespace armnnOpaqueDelegate