//
// Copyright © 2023 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include <OpaqueDelegateUtils.hpp>

#include <tensorflow/lite/builtin_ops.h>
#include <tensorflow/lite/c/builtin_op_data.h>
#include <tensorflow/lite/c/common.h>
#include <tensorflow/lite/minimal_logging.h>

namespace armnnOpaqueDelegate
{

TfLiteStatus VisitBatchToSpaceNdOperator(DelegateData& delegateData,
                                         TfLiteOpaqueContext* tfLiteContext,
                                         TfLiteOpaqueNode* tfLiteNode,
                                         int nodeIndex,
                                         int32_t operatorCode)
{
    TF_LITE_ENSURE_STATUS(ValidateNumInputs(tfLiteContext, tfLiteNode, 3, nodeIndex));
    TF_LITE_ENSURE_STATUS(ValidateNumOutputs(tfLiteContext, tfLiteNode, 1, nodeIndex));

    int numInputs = 3;
    const int* inputTensors;
    if (TfLiteOpaqueNodeInputs(tfLiteNode, &inputTensors, &numInputs) != kTfLiteOk)
    {
        TF_LITE_OPAQUE_MAYBE_KERNEL_LOG(
                tfLiteContext,
                "TfLiteArmnnOpaqueDelegate: Unable to gather input tensor indices from node #%d: ",
                nodeIndex);
        return kTfLiteError;
    }

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

    const TfLiteOpaqueTensor* tfLiteInputTensor = TfLiteOpaqueContextGetOpaqueTensor(tfLiteContext, inputTensors[0]);
    if (!IsValid(tfLiteContext, tfLiteInputTensor, operatorCode, nodeIndex))
    {
        return kTfLiteError;
    }
    const TfLiteOpaqueTensor* tfLiteBlockShapeTensor = TfLiteOpaqueContextGetOpaqueTensor(tfLiteContext,
                                                                                         inputTensors[1]);
    if (!IsValid(tfLiteContext, tfLiteBlockShapeTensor, operatorCode, nodeIndex))
    {
        return kTfLiteError;
    }
    const TfLiteOpaqueTensor* tfLiteCropsTensor = TfLiteOpaqueContextGetOpaqueTensor(tfLiteContext, inputTensors[2]);
    if (!IsValid(tfLiteContext, tfLiteCropsTensor, operatorCode, nodeIndex))
    {
        return kTfLiteError;
    }
    const TfLiteOpaqueTensor* tfLiteOutputTensor = TfLiteOpaqueContextGetOpaqueTensor(tfLiteContext,
                                                                                     outputTensors[0]);
    if (!IsValid(tfLiteContext, tfLiteOutputTensor, operatorCode, nodeIndex))
    {
        return kTfLiteError;
    }

    const armnn::TensorInfo& inputTensorInfo      = GetTensorInfoForTfLiteOpaqueTensor(tfLiteInputTensor);
    const armnn::TensorInfo& blockShapeTensorInfo = GetTensorInfoForTfLiteOpaqueTensor(tfLiteBlockShapeTensor);
    const armnn::TensorInfo& cropsTensorInfo      = GetTensorInfoForTfLiteOpaqueTensor(tfLiteCropsTensor);
    const armnn::TensorInfo& outputTensorInfo     = GetTensorInfoForTfLiteOpaqueTensor(tfLiteOutputTensor, true);


    // Copy memory into block and crops
    std::vector<unsigned int> blockShape(blockShapeTensorInfo.GetNumElements());
    ::memcpy(blockShape.data(), TfLiteOpaqueTensorData(tfLiteBlockShapeTensor), blockShapeTensorInfo.GetNumBytes());

    std::vector<unsigned int> cropsVector(cropsTensorInfo.GetNumElements());
    std::memcpy(cropsVector.data(), TfLiteOpaqueTensorData(tfLiteCropsTensor), cropsTensorInfo.GetNumBytes());

    size_t step = 2;
    std::vector<std::pair<unsigned int, unsigned int>> crops;
    for (unsigned int i = 0; i < cropsTensorInfo.GetNumElements() / step; ++i)
    {
        crops.emplace_back(cropsVector[i * step], cropsVector[i * step + 1]);
    }

    // Make a descriptor
    armnn::BatchToSpaceNdDescriptor descriptor;
    descriptor.m_BlockShape = blockShape;
    descriptor.m_Crops = crops;
    descriptor.m_DataLayout = armnn::DataLayout::NHWC;

    // Check if supported
    bool isSupported = false;
    armnn::BackendId setBackend;
    auto validateFunc = [&](const armnn::TensorInfo& outputTensorInfo, bool& isSupported)
    {
        FORWARD_LAYER_OPAQUE_SUPPORT_FUNC("BATCH_TO_SPACE_ND",
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
    return Connect(layer, tfLiteContext, tfLiteNode, delegateData);
}

TfLiteStatus VisitSpaceToBatchNdOperator(DelegateData& delegateData,
                                         TfLiteOpaqueContext* tfLiteContext,
                                         TfLiteOpaqueNode* tfLiteNode,
                                         int nodeIndex,
                                         int32_t operatorCode)
{
    TF_LITE_ENSURE_STATUS(ValidateNumInputs(tfLiteContext, tfLiteNode, 3, nodeIndex));
    TF_LITE_ENSURE_STATUS(ValidateNumOutputs(tfLiteContext, tfLiteNode, 1, nodeIndex));

    int numInputs = 3;
    const int* inputTensors;
    if (TfLiteOpaqueNodeInputs(tfLiteNode, &inputTensors, &numInputs) != kTfLiteOk)
    {
        TF_LITE_OPAQUE_MAYBE_KERNEL_LOG(
                tfLiteContext,
                "TfLiteArmnnOpaqueDelegate: Unable to gather input tensor indices from node #%d: ",
                nodeIndex);
        return kTfLiteError;
    }

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

    const TfLiteOpaqueTensor* tfLiteInputTensor  = TfLiteOpaqueContextGetOpaqueTensor(tfLiteContext,inputTensors[0]);
    if (!IsValid(tfLiteContext, tfLiteInputTensor, operatorCode, nodeIndex))
    {
        return kTfLiteError;
    }
    const TfLiteOpaqueTensor* tfLiteBlockShapeTensor = TfLiteOpaqueContextGetOpaqueTensor(tfLiteContext,
                                                                                          inputTensors[1]);
    if (!IsValid(tfLiteContext, tfLiteBlockShapeTensor, operatorCode, nodeIndex))
    {
        return kTfLiteError;
    }
    const TfLiteOpaqueTensor* tfLitePadListTensor = TfLiteOpaqueContextGetOpaqueTensor(tfLiteContext,
                                                                                       inputTensors[2]);
    if (!IsValid(tfLiteContext, tfLitePadListTensor, operatorCode, nodeIndex))
    {
        return kTfLiteError;
    }
    const TfLiteOpaqueTensor* tfLiteOutputTensor = TfLiteOpaqueContextGetOpaqueTensor(tfLiteContext,
                                                                                      outputTensors[0]);
    if (!IsValid(tfLiteContext, tfLiteOutputTensor, operatorCode, nodeIndex))
    {
        return kTfLiteError;
    }

    const armnn::TensorInfo& inputTensorInfo        = GetTensorInfoForTfLiteOpaqueTensor(tfLiteInputTensor);
    const armnn::TensorInfo& blockShapeTensorInfo   = GetTensorInfoForTfLiteOpaqueTensor(tfLiteBlockShapeTensor);
    const armnn::TensorInfo& padListTensorInfo      = GetTensorInfoForTfLiteOpaqueTensor(tfLitePadListTensor);
    const armnn::TensorInfo& outputTensorInfo       = GetTensorInfoForTfLiteOpaqueTensor(tfLiteOutputTensor, true);

    std::vector<unsigned int> blockShape(blockShapeTensorInfo.GetNumElements());
    std::memcpy(blockShape.data(),
                TfLiteOpaqueTensorData(tfLiteBlockShapeTensor),
                blockShapeTensorInfo.GetNumBytes());

    std::vector<unsigned int> padListVector(padListTensorInfo.GetNumElements());
    std::memcpy(padListVector.data(),
                TfLiteOpaqueTensorData(tfLitePadListTensor),
                padListTensorInfo.GetNumBytes());

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
        FORWARD_LAYER_OPAQUE_SUPPORT_FUNC("SPACE_TO_BATCH_ND",
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
    return Connect(layer, tfLiteContext, tfLiteNode, delegateData);
}

} // namespace