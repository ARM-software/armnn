//
// Copyright © 2023 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include <OpaqueDelegateUtils.hpp>

namespace armnnOpaqueDelegate
{

TfLiteStatus VisitTransposeOperator(DelegateData& delegateData,
                                    TfLiteOpaqueContext* tfLiteContext,
                                    TfLiteOpaqueNode* tfLiteNode,
                                    int nodeIndex,
                                    int32_t tfliteTransposeOperatorCode)
{
    TF_LITE_ENSURE_STATUS(ValidateNumInputs(tfLiteContext, tfLiteNode, 2, nodeIndex));
    TF_LITE_ENSURE_STATUS(ValidateNumOutputs(tfLiteContext, tfLiteNode, 1, nodeIndex));

    // Gather input indices and use to get input tensor.
    const int* inputTensors;
    int numInputs;
    if (TfLiteOpaqueNodeInputs(tfLiteNode, &inputTensors, &numInputs) != kTfLiteOk)
    {
        TF_LITE_OPAQUE_MAYBE_KERNEL_LOG(
                tfLiteContext,
                "TfLiteArmnnOpaqueDelegate: Unable to gather input tensor indices from node #%d: ",
                nodeIndex);
        return kTfLiteError;
    }
    const TfLiteOpaqueTensor* tfLiteInputTensor0 = TfLiteOpaqueContextGetOpaqueTensor(tfLiteContext, inputTensors[0]);
    if (!IsValid(tfLiteContext, tfLiteInputTensor0, tfliteTransposeOperatorCode, nodeIndex))
    {
        return kTfLiteError;
    }
    const TfLiteOpaqueTensor* tfLiteInputTensor1 = TfLiteOpaqueContextGetOpaqueTensor(tfLiteContext, inputTensors[1]);
    if (!IsValid(tfLiteContext, tfLiteInputTensor1, tfliteTransposeOperatorCode, nodeIndex))
    {
        return kTfLiteError;
    }

    // Gather output indices and use to get output tensors.
    const int* outputTensors;
    int numOutputs;
    if (TfLiteOpaqueNodeOutputs(tfLiteNode, &outputTensors, &numOutputs) != kTfLiteOk)
    {
        TF_LITE_OPAQUE_MAYBE_KERNEL_LOG(
                tfLiteContext,
                "TfLiteArmnnOpaqueDelegate: Unable to gather output tensor indices from node #%d: ",
                nodeIndex);
        return kTfLiteError;
    }

    const TfLiteOpaqueTensor* tfLiteOutputTensor = TfLiteOpaqueContextGetOpaqueTensor(tfLiteContext, outputTensors[0]);
    if (!IsValid(tfLiteContext, tfLiteOutputTensor, tfliteTransposeOperatorCode, nodeIndex))
    {
        return kTfLiteError;
    }

    const armnn::TensorInfo& inputTensorInfo0 = GetTensorInfoForTfLiteOpaqueTensor(tfLiteInputTensor0);
    const armnn::TensorInfo& outputTensorInfo = GetTensorInfoForTfLiteOpaqueTensor(tfLiteOutputTensor, true);

    auto* permTensorDataPtr = static_cast<int32_t*>(TfLiteOpaqueTensorData(tfLiteInputTensor1));
    unsigned int numEl = TfLiteOpaqueTensorDim(tfLiteInputTensor1, 0);

    if ( numEl > static_cast<int>(armnn::MaxNumOfTensorDimensions) )
    {
        return kTfLiteError;
    }

    // Ensure only single dimension to the permutation tensor
    if ( TfLiteOpaqueTensorNumDims(tfLiteInputTensor1) != 1 )
    {
        return kTfLiteError;
    }

    armnn::TransposeDescriptor descriptor(armnn::PermutationVector(
            reinterpret_cast<const armnn::PermutationVector::ValueType *> (permTensorDataPtr),
            static_cast<armnn::PermutationVector::SizeType>(numEl)));

    bool isSupported = false;
    armnn::BackendId setBackend;
    auto validateFunc = [&](const armnn::TensorInfo& outputTensorInfo, bool& isSupported)
    {
        FORWARD_LAYER_OPAQUE_SUPPORT_FUNC("TRANSPOSE",
                                          tfLiteContext,
                                          IsTransposeSupported,
                                          delegateData.m_Backends,
                                          isSupported,
                                          setBackend,
                                          inputTensorInfo0,
                                          outputTensorInfo,
                                          descriptor);
    };

    if (!delegateData.m_Network)
    {
        validateFunc(outputTensorInfo, isSupported);
        return isSupported ? kTfLiteOk : kTfLiteError;
    }

    auto layerName = GetName(armnn::LayerType::Transpose, nodeIndex);
    armnn::IConnectableLayer* transposeLayer = delegateData.m_Network->AddTransposeLayer(descriptor, layerName.c_str());
    transposeLayer->SetBackendId(setBackend);
    ARMNN_ASSERT(transposeLayer != nullptr);
    // Permutation vector given to descriptor object
    if (transposeLayer->GetNumInputSlots() != 1)
    {
        return kTfLiteError;
    }

    armnn::IOutputSlot& outputSlot = transposeLayer->GetOutputSlot(0);
    outputSlot.SetTensorInfo(outputTensorInfo);

    // try to connect the Constant Inputs if there are any
    if (ProcessInputs(transposeLayer, delegateData, tfLiteContext, tfLiteNode, nodeIndex) != kTfLiteOk)
    {
        return kTfLiteError;
    }

    return Connect(transposeLayer, tfLiteContext, tfLiteNode, delegateData);
}
} // namespace armnnOpaqueDelegate
