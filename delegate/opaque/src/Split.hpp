//
// Copyright © 2023 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include <OpaqueDelegateUtils.hpp>
#include <DelegateUtils.hpp>

#include <algorithm>
#include <iterator>
#include <vector>

namespace armnnOpaqueDelegate
{

constexpr unsigned int MaxNumOfTensorDimensions = 5U;

TfLiteStatus VisitSplitOperator(DelegateData& delegateData,
                                TfLiteOpaqueContext* tfLiteContext,
                                TfLiteOpaqueNode* tfLiteNode,
                                int nodeIndex,
                                int32_t tfLiteSplitOperatorCode)
{
    TF_LITE_ENSURE_STATUS(ValidateNumInputs(tfLiteContext, tfLiteNode, 2, nodeIndex));

    auto* splitParameters = reinterpret_cast<TfLiteSplitParams*>(TfLiteOpaqueNodeGetBuiltinData(tfLiteNode));
    int numSplits =  NonNegative(splitParameters->num_splits, nodeIndex);

    TF_LITE_ENSURE_STATUS(ValidateNumOutputs(tfLiteContext, tfLiteNode, numSplits, nodeIndex));

    // Gather input indices and use to get Axis tensor.
    const int* inputTensors;
    auto numInputs = TfLiteOpaqueNodeNumberOfInputs(tfLiteNode);
    if (TfLiteOpaqueNodeInputs(tfLiteNode, &inputTensors, &numInputs) != kTfLiteOk)
    {
        TF_LITE_OPAQUE_MAYBE_KERNEL_LOG(
                tfLiteContext,
                "TfLiteArmnnOpaqueDelegate: Unable to gather input tensor indices from node #%d: ",
                nodeIndex);
        return kTfLiteError;
    }

    const TfLiteOpaqueTensor* tfLiteAxisTensor = TfLiteOpaqueContextGetOpaqueTensor(tfLiteContext, inputTensors[0]);
    if (!IsValid(tfLiteContext, tfLiteAxisTensor, tfLiteSplitOperatorCode, nodeIndex))
    {
        return kTfLiteError;
    }

    // Use input indices to get input tensor.
    const TfLiteOpaqueTensor* tfLiteInputTensor = TfLiteOpaqueContextGetOpaqueTensor(tfLiteContext, inputTensors[1]);
    if (!IsValid(tfLiteContext, tfLiteInputTensor, tfLiteSplitOperatorCode, nodeIndex))
    {
        return kTfLiteError;
    }

    // Gather output indices and use to get output tensors.
    const int* outputTensors;
    if (TfLiteOpaqueNodeOutputs(tfLiteNode, &outputTensors, &numSplits) != kTfLiteOk)
    {
        TF_LITE_OPAQUE_MAYBE_KERNEL_LOG(
                tfLiteContext,
                "TfLiteArmnnOpaqueDelegate: Unable to gather output tensor indices from node #%d: ",
                nodeIndex);
        return kTfLiteError;
    }

    const armnn::TensorInfo& inputTensorInfo = GetTensorInfoForTfLiteOpaqueTensor(tfLiteInputTensor);

    if (GetTensorInfoForTfLiteOpaqueTensor(tfLiteAxisTensor).GetNumElements() != 1)
    {
        return kTfLiteError;
    }

    auto* axisTensorDataPtr = static_cast<uint32_t*>(TfLiteOpaqueTensorData(tfLiteAxisTensor));
    std::vector<int32_t> axisTensorData(axisTensorDataPtr, axisTensorDataPtr + 1);
    int32_t axis = axisTensorData[0];

    auto inputDimensions = static_cast<int32_t>(inputTensorInfo.GetNumDimensions());
    if (((axis < -inputDimensions) && (axis < 0)) || ((axis >= inputDimensions) && (axis > 0)))
    {
        // Square bracket denotes inclusive n while parenthesis denotes exclusive n
        // E.g. Rank 4 tensor can have axis in range [-4, 3)
        // -1 == 3, -2 == 2, -3 == 1, -4 == 0
        TF_LITE_OPAQUE_MAYBE_KERNEL_LOG(
                tfLiteContext,
                "TfLiteOpaqueArmnnDelegate: Operation has invalid axis: #%d. "
                "Axis must be in range [-n, n) in node #%d:",
                axis, nodeIndex);
    }
    const unsigned int splitDim = ComputeWrappedIndex(axis, inputTensorInfo.GetNumDimensions());

    std::vector<armnn::TensorInfo> outputs;
    for (int i = 0; i < numSplits; ++i)
    {
        const TfLiteOpaqueTensor* tfLiteOutputTensor =
                TfLiteOpaqueContextGetOpaqueTensor(tfLiteContext, outputTensors[i]);
        if (!IsValid(tfLiteContext, tfLiteOutputTensor, tfLiteSplitOperatorCode, nodeIndex))
        {
            return kTfLiteError;
        }
        outputs.push_back(GetTensorInfoForTfLiteOpaqueTensor(tfLiteOutputTensor, true));
    }
    const std::vector<std::reference_wrapper<armnn::TensorInfo>> outputTensorInfos(outputs.begin(), outputs.end());

    auto inputDimSize = inputTensorInfo.GetNumDimensions();
    if (inputDimSize > MaxNumOfTensorDimensions)
    {
        TF_LITE_OPAQUE_MAYBE_KERNEL_LOG(
                tfLiteContext,
                "TfLiteOpaqueArmnnDelegate: The number of dimensions: #%d for input tensors of the split op cannot be "
                "greater than #%d in node #%d: ",
                inputDimSize, MaxNumOfTensorDimensions, nodeIndex);
        return kTfLiteError;
    }

    std::vector<unsigned int> splitterDimSizes(inputDimSize);

    // Add current input shape to splitterDimSizes
    for (unsigned int i = 0; i < inputDimSize; ++i)
    {
        splitterDimSizes[i] = inputTensorInfo.GetShape()[i];
    }

    if (splitterDimSizes[splitDim] % numSplits != 0)
    {
        TF_LITE_OPAQUE_MAYBE_KERNEL_LOG(
                tfLiteContext,
                "TfLiteOpaqueArmnnDelegate: Number of splits #%d must evenly divide the dimension #%d in node #%d: ",
                numSplits, splitterDimSizes[splitDim], nodeIndex);
        return kTfLiteError;
    }
    splitterDimSizes[splitDim] /= numSplits;

    armnn::SplitterDescriptor splitDescriptor(numSplits, inputDimSize);
    splitDescriptor.SetAxis(axis);

    for (int j = 0; j < numSplits; ++j)
    {
        // Set the size of the views.
        for (unsigned int dimIdx = 0; dimIdx < splitterDimSizes.size(); ++dimIdx)
        {
            splitDescriptor.SetViewSize(j, dimIdx, splitterDimSizes[dimIdx]);
        }
        splitDescriptor.SetViewOriginCoord(j, splitDim, splitterDimSizes[splitDim] * j);
    }

    armnn::BackendId setBackend;
    if (!delegateData.m_Network)
    {
        // Check if supported
        bool isSupported = false;
        FORWARD_LAYER_OPAQUE_SUPPORT_FUNC("SPLIT",
                                   tfLiteContext,
                                   IsSplitterSupported,
                                   delegateData.m_Backends,
                                   isSupported,
                                   setBackend,
                                   inputTensorInfo,
                                   outputTensorInfos,
                                   splitDescriptor);
        return isSupported ? kTfLiteOk : kTfLiteError;
    }

    auto layerName = GetName(armnn::LayerType::Splitter, nodeIndex);
    armnn::IConnectableLayer* layer = delegateData.m_Network->AddSplitterLayer(splitDescriptor, layerName.c_str());
    layer->SetBackendId(setBackend);
    ARMNN_ASSERT(layer != nullptr);

    for (unsigned int k = 0; k < layer->GetNumOutputSlots(); ++k)
    {
        layer->GetOutputSlot(k).SetTensorInfo(outputs[k]);
    }

    // Connect the input slots
    delegateData.m_OutputSlotForNode[inputTensors[1]]->Connect(layer->GetInputSlot(0));

    if(numSplits != static_cast<int>(layer->GetNumOutputSlots()))
    {
        TF_LITE_OPAQUE_MAYBE_KERNEL_LOG(
                tfLiteContext,
                "TfLiteOpaqueArmnnDelegate: Expected number of splits #%d does not "
                "match the number of output slots #%d in node #%d: ",
                numSplits, layer->GetNumOutputSlots(), nodeIndex);
        return kTfLiteError;
    }

    // Prepare output slots
    for (unsigned int outputIndex = 0; outputIndex < layer->GetNumOutputSlots(); ++outputIndex)
    {
        armnn::IOutputSlot& outputSlot = layer->GetOutputSlot(outputIndex);
        delegateData.m_OutputSlotForNode[
                static_cast<unsigned long>(outputTensors[outputIndex])] = &outputSlot;
    }
    return kTfLiteOk;
}

TfLiteStatus VisitSplitVOperator(DelegateData& delegateData,
                                 TfLiteOpaqueContext* tfLiteContext,
                                 TfLiteOpaqueNode* tfLiteNode,
                                 int nodeIndex,
                                 int32_t tfLiteSplitVOperatorCode)
{

    TF_LITE_ENSURE_STATUS(ValidateNumInputs(tfLiteContext, tfLiteNode, 3, nodeIndex));

    const int* inputTensors;
    auto numInputs = TfLiteOpaqueNodeNumberOfInputs(tfLiteNode);
    if (TfLiteOpaqueNodeInputs(tfLiteNode, &inputTensors, &numInputs) != kTfLiteOk)
    {
        TF_LITE_OPAQUE_MAYBE_KERNEL_LOG(
                tfLiteContext,
                "TfLiteArmnnOpaqueDelegate: Unable to gather input tensor indices from node #%d: ",
                nodeIndex);
        return kTfLiteError;
    }

    const TfLiteOpaqueTensor* tfLiteInputTensor = TfLiteOpaqueContextGetOpaqueTensor(tfLiteContext, inputTensors[0]);
    if (!IsValid(tfLiteContext, tfLiteInputTensor, tfLiteSplitVOperatorCode, nodeIndex))
    {
        return kTfLiteError;
    }

    const TfLiteOpaqueTensor* tfLiteSplitsTensor = TfLiteOpaqueContextGetOpaqueTensor(tfLiteContext, inputTensors[1]);
    if (!IsValid(tfLiteContext, tfLiteSplitsTensor, tfLiteSplitVOperatorCode, nodeIndex))
    {
        return kTfLiteError;
    }

    const TfLiteOpaqueTensor* tfLiteAxisTensor = TfLiteOpaqueContextGetOpaqueTensor(tfLiteContext, inputTensors[2]);
    if (!IsValid(tfLiteContext, tfLiteAxisTensor, tfLiteSplitVOperatorCode, nodeIndex))
    {
        return kTfLiteError;
    }

    const armnn::TensorInfo& inputTensorInfo = GetTensorInfoForTfLiteOpaqueTensor(tfLiteInputTensor);
    const armnn::TensorInfo& splitsTensorInfo = GetTensorInfoForTfLiteOpaqueTensor(tfLiteSplitsTensor);

    if (splitsTensorInfo.GetNumDimensions() != 1)
    {
        return kTfLiteError;
    }

    if (GetTensorInfoForTfLiteOpaqueTensor(tfLiteAxisTensor).GetNumElements() != 1)
    {
        return kTfLiteError;
    }

    auto* axisTensorDataPtr = static_cast<uint32_t*>(TfLiteOpaqueTensorData(tfLiteAxisTensor));
    std::vector<int32_t> axisTensorData(axisTensorDataPtr, axisTensorDataPtr + 1);
    int32_t axis = axisTensorData[0];

    auto inputDimensions = static_cast<int32_t>(inputTensorInfo.GetNumDimensions());
    if (((axis < -inputDimensions) && (axis < 0)) || ((axis >= inputDimensions) && (axis > 0)))
    {
        TF_LITE_OPAQUE_MAYBE_KERNEL_LOG(
                tfLiteContext,
                "TfLiteOpaqueArmnnDelegate: Operation has invalid axis: #%d. "
                "Axis must be in range [-n, n) in node #%d:",
                axis, nodeIndex);
    }
    const unsigned int splitDim = ComputeWrappedIndex(axisTensorData[0], inputTensorInfo.GetNumDimensions());

    auto* splitVParameters = reinterpret_cast<TfLiteSplitVParams*>(TfLiteOpaqueNodeGetBuiltinData(tfLiteNode));
    int numSplits = 0;
    if (splitVParameters)
    {
        numSplits = NonNegative(splitVParameters->num_splits, nodeIndex);
    }
    else
    {
        numSplits = splitsTensorInfo.GetNumElements();
    }

    if (numSplits <= 0)
    {
        TF_LITE_OPAQUE_MAYBE_KERNEL_LOG(
                tfLiteContext,
                "TfLiteOpaqueArmnnDelegate: Invalid number of splits %d  in node #%d",
                numSplits, nodeIndex);
        return kTfLiteError;
    }

    TF_LITE_ENSURE_STATUS(ValidateNumOutputs(tfLiteContext, tfLiteNode, numSplits, nodeIndex));

    // Gather output indices and use to get output tensors.
    const int* outputTensors;
    if (TfLiteOpaqueNodeOutputs(tfLiteNode, &outputTensors, &numSplits) != kTfLiteOk)
    {
        TF_LITE_OPAQUE_MAYBE_KERNEL_LOG(
                tfLiteContext,
                "TfLiteArmnnOpaqueDelegate: Unable to gather output tensor indices from node #%d: ",
                nodeIndex);
        return kTfLiteError;
    }
    std::vector<armnn::TensorInfo> outputs;
    for (int i = 0; i < numSplits; ++i)
    {
        const TfLiteOpaqueTensor* tfLiteOutputTensor =
                TfLiteOpaqueContextGetOpaqueTensor(tfLiteContext, outputTensors[i]);
        if (!IsValid(tfLiteContext, tfLiteOutputTensor, tfLiteSplitVOperatorCode, nodeIndex))
        {
            return kTfLiteError;
        }
        outputs.push_back(GetTensorInfoForTfLiteOpaqueTensor(tfLiteOutputTensor, true));
    }
    const std::vector<std::reference_wrapper<armnn::TensorInfo>> outputTensorInfos(outputs.begin(), outputs.end());

    auto inputDimSize = inputTensorInfo.GetNumDimensions();
    if (inputDimSize > MaxNumOfTensorDimensions)
    {
        TF_LITE_OPAQUE_MAYBE_KERNEL_LOG(
                tfLiteContext,
                "TfLiteOpaqueArmnnDelegate: The number of dimensions: #%d for input tensors of the split op cannot be "
                "greater than #%d in node #%d: ",
                inputDimSize, MaxNumOfTensorDimensions, nodeIndex);
        return kTfLiteError;
    }

    std::vector<int32_t> splitsTensorData(numSplits);
    std::memcpy(splitsTensorData.data(), TfLiteOpaqueTensorData(tfLiteSplitsTensor), splitsTensorInfo.GetNumBytes());


    unsigned int index         = 0;
    unsigned int inferredIndex = 0;
    int numberOfInferred       = 0;
    int splitSum = 0;

    for (auto splitData : splitsTensorData)
    {
        if (splitData < 0)
        {
            ++numberOfInferred;
            inferredIndex = index;
        }
        else
        {
            splitSum += splitData;
        }
        ++index;
    }

    // Check for inferred axis
    if (numberOfInferred == 0)
    {
        if (splitSum != armnn::numeric_cast<int>(inputTensorInfo.GetShape()[splitDim]))
        {
            TF_LITE_OPAQUE_MAYBE_KERNEL_LOG(
                    tfLiteContext,
                    "TfLiteOpaqueArmnnDelegate: SplitV split_sizes does not sum to the dimension "
                    "of value along split_dim in node #%d",
                    nodeIndex);
            return kTfLiteError;
        }
    }
    else if (numberOfInferred == 1)
    {
        splitsTensorData[inferredIndex] = armnn::numeric_cast<int>(inputTensorInfo.GetShape()[splitDim]) - splitSum;
    }
    else
    {
        TF_LITE_OPAQUE_MAYBE_KERNEL_LOG(
                tfLiteContext,
                "TfLiteOpaqueArmnnDelegate: SplitV cannot infer split size for "
                "more than one split in node #%d",
                nodeIndex);
        return kTfLiteError;
    }

    armnn::SplitterDescriptor splitDescriptor(numSplits, inputDimSize);
    splitDescriptor.SetAxis(axis);
    unsigned int accumSplit = 0;

    for (int j = 0; j < numSplits; ++j)
    {
        unsigned int splitSize = armnn::numeric_cast<unsigned int>(splitsTensorData[j]);

        // Set the size of the views.
        for (unsigned int dimIdx = 0; dimIdx < inputTensorInfo.GetNumDimensions(); ++dimIdx)
        {
            unsigned int dimSize = inputTensorInfo.GetShape()[dimIdx];
            if (dimIdx == splitDim)
            {
                dimSize = splitSize;
            }
            splitDescriptor.SetViewSize(j, dimIdx, dimSize);
        }

        splitDescriptor.SetViewOriginCoord(j, splitDim, accumSplit);
        accumSplit += splitSize;
    }

    armnn::BackendId setBackend;
    if (!delegateData.m_Network)
    {
        // Check if supported
        bool isSupported = false;
        FORWARD_LAYER_OPAQUE_SUPPORT_FUNC("SPLITV",
                                          tfLiteContext,
                                          IsSplitterSupported,
                                          delegateData.m_Backends,
                                          isSupported,
                                          setBackend,
                                          inputTensorInfo,
                                          outputTensorInfos,
                                          splitDescriptor);
        return isSupported ? kTfLiteOk : kTfLiteError;
    }

    auto layerName = GetName(armnn::LayerType::Splitter, nodeIndex);
    armnn::IConnectableLayer* layer = delegateData.m_Network->AddSplitterLayer(splitDescriptor, layerName.c_str());
    layer->SetBackendId(setBackend);
    ARMNN_ASSERT(layer != nullptr);

    for (unsigned int k = 0; k < layer->GetNumOutputSlots(); ++k)
    {
        layer->GetOutputSlot(k).SetTensorInfo(outputs[k]);
    }

    // try to connect the Constant Inputs if there are any
    if (ProcessInputs(layer, delegateData, tfLiteContext, tfLiteNode, nodeIndex) != kTfLiteOk)
    {
        return kTfLiteError;
    }

    // Connect
    return Connect(layer, tfLiteContext, tfLiteNode, delegateData);
}

} // namespace armnnOpaqueDelegate