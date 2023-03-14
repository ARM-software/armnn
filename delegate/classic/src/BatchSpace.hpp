//
// Copyright © 2022-2023 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include <tensorflow/lite/builtin_ops.h>
#include <tensorflow/lite/c/builtin_op_data.h>
#include <tensorflow/lite/c/common.h>
#include <tensorflow/lite/minimal_logging.h>

namespace armnnDelegate
{

TfLiteStatus VisitBatchToSpaceNdOperator(DelegateData& delegateData,
                                         TfLiteContext* tfLiteContext,
                                         TfLiteNode* tfLiteNode,
                                         int nodeIndex,
                                         int32_t operatorCode)
{
    TF_LITE_ENSURE_STATUS(ValidateNumInputs(tfLiteContext, tfLiteNode, 3, nodeIndex));
    TF_LITE_ENSURE_STATUS(ValidateNumOutputs(tfLiteContext, tfLiteNode, 1, nodeIndex));

    const TfLiteTensor* tfLiteTensors = tfLiteContext->tensors;
    const TfLiteTensor& tfLiteInputTensor = tfLiteTensors[tfLiteNode->inputs->data[0]];
    if (!IsValid(tfLiteContext, tfLiteInputTensor, operatorCode, nodeIndex))
    {
        return kTfLiteError;
    }

    const TfLiteTensor& tfLiteBlockShapeTensor = tfLiteTensors[tfLiteNode->inputs->data[1]];
    if (!IsValid(tfLiteContext, tfLiteBlockShapeTensor, operatorCode, nodeIndex))
    {
        return kTfLiteError;
    }

    const TfLiteTensor& tfLiteCropsTensor = tfLiteTensors[tfLiteNode->inputs->data[2]];
    if (!IsValid(tfLiteContext, tfLiteCropsTensor, operatorCode, nodeIndex))
    {
        return kTfLiteError;
    }

    const TfLiteTensor& tfLiteOutputTensor = tfLiteTensors[tfLiteNode->outputs->data[0]];
    if (!IsValid(tfLiteContext, tfLiteOutputTensor, operatorCode, nodeIndex))
    {
        return kTfLiteError;
    }

    const armnn::TensorInfo& inputTensorInfo      = GetTensorInfoForTfLiteTensor(tfLiteInputTensor);
    const armnn::TensorInfo& blockShapeTensorInfo = GetTensorInfoForTfLiteTensor(tfLiteBlockShapeTensor);
    const armnn::TensorInfo& cropsTensorInfo      = GetTensorInfoForTfLiteTensor(tfLiteCropsTensor);
    const armnn::TensorInfo& outputTensorInfo     = GetTensorInfoForTfLiteTensor(tfLiteOutputTensor, true);

    std::vector<unsigned int> blockShape(blockShapeTensorInfo.GetNumElements());
    ::memcpy(blockShape.data(), tfLiteBlockShapeTensor.data.data, blockShapeTensorInfo.GetNumBytes());

    std::vector<unsigned int> cropsVector(cropsTensorInfo.GetNumElements());
    std::memcpy(cropsVector.data(), tfLiteCropsTensor.data.data, cropsTensorInfo.GetNumBytes());

    size_t step = 2;
    std::vector<std::pair<unsigned int, unsigned int>> crops;
    for (unsigned int i = 0; i < cropsTensorInfo.GetNumElements() / step; ++i)
    {
        crops.emplace_back(cropsVector[i * step], cropsVector[i * step + 1]);
    }

    armnn::BatchToSpaceNdDescriptor descriptor;
    descriptor.m_BlockShape = blockShape;
    descriptor.m_Crops = crops;
    descriptor.m_DataLayout = armnn::DataLayout::NHWC;

    // Check if supported
    bool isSupported = false;
    armnn::BackendId setBackend;
    auto validateFunc = [&](const armnn::TensorInfo& outputTensorInfo, bool& isSupported)
    {
        FORWARD_LAYER_SUPPORT_FUNC("BATCH_TO_SPACE_ND",
                                   tfLiteContext,
                                   IsBatchToSpaceNdSupported,
                                   delegateData.m_Backends,
                                   isSupported,
                                   setBackend,
                                   inputTensorInfo,
                                   outputTensorInfo,
                                   descriptor);
    };

    // If the m_Network is a nullptr, this signals that a prerequisite TfLite callback is required to clarify the
    // support for the operator
    // If supported, VisitBatchToSpaceNdOperator will be called again to add the layer to the network as seen below
    if (!delegateData.m_Network)
    {
        validateFunc(outputTensorInfo, isSupported);
        return isSupported ? kTfLiteOk : kTfLiteError;
    }

    // Add a BatchToSpace layer
    armnn::IConnectableLayer* layer = delegateData.m_Network->AddBatchToSpaceNdLayer(descriptor);
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

TfLiteStatus VisitSpaceToBatchNdOperator(DelegateData& delegateData,
                                         TfLiteContext* tfLiteContext,
                                         TfLiteNode* tfLiteNode,
                                         int nodeIndex,
                                         int32_t operatorCode)
{
    TF_LITE_ENSURE_STATUS(ValidateNumInputs(tfLiteContext, tfLiteNode, 3, nodeIndex));
    TF_LITE_ENSURE_STATUS(ValidateNumOutputs(tfLiteContext, tfLiteNode, 1, nodeIndex));

    const TfLiteTensor* tfLiteTensors = tfLiteContext->tensors;
    const TfLiteTensor& tfLiteInputTensor = tfLiteTensors[tfLiteNode->inputs->data[0]];
    if (!IsValid(tfLiteContext, tfLiteInputTensor, operatorCode, nodeIndex))
    {
        return kTfLiteError;
    }

    const TfLiteTensor& tfLiteBlockShapeTensor = tfLiteTensors[tfLiteNode->inputs->data[1]];
    if (!IsValid(tfLiteContext, tfLiteBlockShapeTensor, operatorCode, nodeIndex))
    {
        return kTfLiteError;
    }

    const TfLiteTensor& tfLitePadListTensor = tfLiteTensors[tfLiteNode->inputs->data[2]];
    if (!IsValid(tfLiteContext, tfLitePadListTensor, operatorCode, nodeIndex))
    {
        return kTfLiteError;
    }

    const TfLiteTensor& tfLiteOutputTensor = tfLiteTensors[tfLiteNode->outputs->data[0]];
    if (!IsValid(tfLiteContext, tfLiteOutputTensor, operatorCode, nodeIndex))
    {
        return kTfLiteError;
    }

    const armnn::TensorInfo& inputTensorInfo      = GetTensorInfoForTfLiteTensor(tfLiteInputTensor);
    const armnn::TensorInfo& blockShapeTensorInfo = GetTensorInfoForTfLiteTensor(tfLiteBlockShapeTensor);
    const armnn::TensorInfo& padListTensorInfo    = GetTensorInfoForTfLiteTensor(tfLitePadListTensor);
    const armnn::TensorInfo& outputTensorInfo     = GetTensorInfoForTfLiteTensor(tfLiteOutputTensor, true);

    std::vector<unsigned int> blockShape(blockShapeTensorInfo.GetNumElements());
    std::memcpy(blockShape.data(), tfLiteBlockShapeTensor.data.data, blockShapeTensorInfo.GetNumBytes());

    std::vector<unsigned int> padListVector(padListTensorInfo.GetNumElements());
    std::memcpy(padListVector.data(), tfLitePadListTensor.data.data, padListTensorInfo.GetNumBytes());

    size_t step = 2;
    std::vector<std::pair<unsigned int, unsigned int>> padList;
    for (unsigned int i = 0; i < padListTensorInfo.GetNumElements() / step; ++i)
    {
        padList.emplace_back(padListVector[i * step], padListVector[i * step + 1]);
    }

    armnn::SpaceToBatchNdDescriptor descriptor;
    descriptor.m_BlockShape = blockShape;
    descriptor.m_PadList = padList;
    descriptor.m_DataLayout = armnn::DataLayout::NHWC;

    // Check if supported
    bool isSupported = false;
    armnn::BackendId setBackend;
    auto validateFunc = [&](const armnn::TensorInfo& outputTensorInfo, bool& isSupported)
    {
        FORWARD_LAYER_SUPPORT_FUNC("SPACE_TO_BATCH_ND",
                                   tfLiteContext,
                                   IsSpaceToBatchNdSupported,
                                   delegateData.m_Backends,
                                   isSupported,
                                   setBackend,
                                   inputTensorInfo,
                                   outputTensorInfo,
                                   descriptor);
    };

    // If the m_Network is a nullptr, this signals that a prerequisite TfLite callback is required to clarify the
    // support for the operator
    // If supported, VisitSpaceToBatchNdOperator will be called again to add the layer to the network as seen below
    if (!delegateData.m_Network)
    {
        validateFunc(outputTensorInfo, isSupported);
        return isSupported ? kTfLiteOk : kTfLiteError;
    }

    // Add a SpaceToBatch layer
    armnn::IConnectableLayer* layer = delegateData.m_Network->AddSpaceToBatchNdLayer(descriptor);
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
