//
// Copyright © 2024 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include <tensorflow/lite/builtin_ops.h>
#include <tensorflow/lite/c/builtin_op_data.h>
#include <tensorflow/lite/c/common.h>
#include <tensorflow/lite/minimal_logging.h>
#include <tensorflow/lite/kernels/internal/tensor_ctypes.h>
#include <tensorflow/lite/schema/schema_generated.h>

namespace armnnDelegate
{
TfLiteStatus ValidateScatterNdOperator(DelegateData& delegateData,
                                       TfLiteContext* tfLiteContext,
                                       const armnn::TensorInfo& indicesInfo,
                                       const armnn::TensorInfo& updatesInfo,
                                       const armnn::TensorInfo& shapeInfo,
                                       const armnn::TensorInfo& outputInfo,
                                       const armnn::ScatterNdDescriptor& descriptor)
{
    bool isSupported = false;
    FORWARD_LAYER_SUPPORT_FUNC("SCATTER_ND",
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
                                    TfLiteContext* tfLiteContext,
                                    TfLiteNode* tfLiteNode,
                                    int nodeIndex,
                                    int32_t scatterNdOperatorCode)
{
    TF_LITE_ENSURE_STATUS(ValidateNumInputs(tfLiteContext, tfLiteNode, 3, nodeIndex));
    TF_LITE_ENSURE_STATUS(ValidateNumOutputs(tfLiteContext, tfLiteNode, 1, nodeIndex));

    const TfLiteTensor* tfLiteTensors = tfLiteContext->tensors;

    // The indices tensor are the positions the data is updated/scattered into
    const TfLiteTensor& tfLiteIndicesTensor = tfLiteTensors[tfLiteNode->inputs->data[0]];
    if (IsDynamicTensor(tfLiteIndicesTensor))
    {
        TF_LITE_MAYBE_KERNEL_LOG(
            tfLiteContext,
            "TfLiteArmnnDelegate: Dynamic input tensors are not supported in operator #%d node #%d: ",
            scatterNdOperatorCode, nodeIndex);
        return kTfLiteError;
    }

    // The updates tensor provides the data which will be updated/scattered into the relevant indices
    const TfLiteTensor& tfLiteUpdatesTensor = tfLiteTensors[tfLiteNode->inputs->data[1]];
    if (IsDynamicTensor(tfLiteUpdatesTensor))
    {
        TF_LITE_MAYBE_KERNEL_LOG(
            tfLiteContext,
            "TfLiteArmnnDelegate: Dynamic input tensors are not supported in operator #%d node #%d: ",
            scatterNdOperatorCode, nodeIndex);
        return kTfLiteError;
    }

    // For tflite scatternd there is no input tensor
    // The shape tensor is a 1D tensor which represents the shape of an input tensor to be filled with zeros
    const TfLiteTensor& tfLiteShapeTensor = tfLiteTensors[tfLiteNode->inputs->data[2]];
    if (IsDynamicTensor(tfLiteUpdatesTensor))
    {
        TF_LITE_MAYBE_KERNEL_LOG(
                tfLiteContext,
                "TfLiteArmnnDelegate: Dynamic input tensors are not supported in operator #%d node #%d: ",
                scatterNdOperatorCode, nodeIndex);
        return kTfLiteError;
    }

    // The output tensor
    const TfLiteTensor& tfLiteOutputTensor = tfLiteTensors[tfLiteNode->outputs->data[0]];
    if (IsDynamicTensor(tfLiteOutputTensor))
    {
        TF_LITE_MAYBE_KERNEL_LOG(
            tfLiteContext,
            "TfLiteArmnnDelegate: Dynamic output tensors are not supported in operator #%d node #%d: ",
            scatterNdOperatorCode, nodeIndex);
        return kTfLiteError;
    }

    const armnn::TensorInfo& indicesTensorInfo = GetTensorInfoForTfLiteTensor(tfLiteIndicesTensor);
    const armnn::TensorInfo& updatesTensorInfo = GetTensorInfoForTfLiteTensor(tfLiteUpdatesTensor);
    const armnn::TensorInfo& shapeTensorInfo   = GetTensorInfoForTfLiteTensor(tfLiteShapeTensor);
    const armnn::TensorInfo& outputTensorInfo  = GetTensorInfoForTfLiteTensor(tfLiteOutputTensor);

    armnn::ScatterNdDescriptor scatterNdDescriptor;
    scatterNdDescriptor.m_Function     = armnn::ScatterNdFunction::Update;
    scatterNdDescriptor.m_InputEnabled = false;
    scatterNdDescriptor.m_Axis         = 0;
    scatterNdDescriptor.m_AxisEnabled  = false;

    // Check output dimensions
    if (shapeTensorInfo.GetShape().GetNumElements() != outputTensorInfo.GetNumDimensions())
    {
        TF_LITE_MAYBE_KERNEL_LOG(
            tfLiteContext,
            "TfLiteArmnnDelegate: Shape tensor number of elements and output tensor dimension differ",
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

    auto layerName = GetLayerName(armnn::LayerType::ScatterNd, nodeIndex);
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

    if (static_cast<unsigned int>(tfLiteNode->outputs->size) != layer->GetNumOutputSlots())
    {
        return kTfLiteError;
    }

    delegateData.m_OutputSlotForNode[tfLiteNode->inputs->data[2]]->Connect(layer->GetInputSlot(0));
    delegateData.m_OutputSlotForNode[tfLiteNode->inputs->data[0]]->Connect(layer->GetInputSlot(1));
    delegateData.m_OutputSlotForNode[tfLiteNode->inputs->data[1]]->Connect(layer->GetInputSlot(2));

    // Prepare output slots
    armnn::IOutputSlot& outputSlot = layer->GetOutputSlot(0);
    delegateData.m_OutputSlotForNode[static_cast<unsigned long>(tfLiteNode->outputs->data[0])] = &outputSlot;

    return kTfLiteOk;
}

} // namespace armnnDelegate