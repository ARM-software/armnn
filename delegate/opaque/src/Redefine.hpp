//
// Copyright © 2023-2024 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//
#pragma once

#include <OpaqueDelegateUtils.hpp>

namespace armnnOpaqueDelegate
{

TfLiteStatus VisitCastOperator(DelegateData& delegateData,
                               TfLiteOpaqueContext* tfLiteContext,
                               TfLiteOpaqueNode* tfLiteNode,
                               int nodeIndex,
                               int32_t operatorCode)
{
    TF_LITE_ENSURE_STATUS(ValidateNumInputs(tfLiteContext, tfLiteNode, 1, nodeIndex));
    TF_LITE_ENSURE_STATUS(ValidateNumOutputs(tfLiteContext, tfLiteNode, 1, nodeIndex));
    int numInputs = 0;
    const int* inputTensors;
    if (TfLiteOpaqueNodeInputs(tfLiteNode, &inputTensors, &numInputs) != kTfLiteOk)
    {
        return kTfLiteError;
    }

    // This layer only has 1 input, so we can directly assign tensor[0] to a new opaque tensor
    const TfLiteOpaqueTensor*
          tfLiteInputTensor = TfLiteOpaqueContextGetOpaqueTensor(tfLiteContext, inputTensors[numInputs-1]);
    if (!IsValid(tfLiteContext, tfLiteInputTensor, operatorCode, nodeIndex))
    {
        return kTfLiteError;
    }

    int numOutputs = 0;
    const int* outputTensors;
    if (TfLiteOpaqueNodeOutputs(tfLiteNode, &outputTensors, &numOutputs) != kTfLiteOk)
    {
        return kTfLiteError;
    }

    // This layer only has 1 output, so we can directly assign tensor[0] to a new opaque tensor
    const TfLiteOpaqueTensor*
          tfLiteOutputTensor = TfLiteOpaqueContextGetOpaqueTensor(tfLiteContext, outputTensors[numOutputs-1]);
    if (!IsValid(tfLiteContext, tfLiteOutputTensor, operatorCode, nodeIndex))
    {
        return kTfLiteError;
    }

    const armnn::TensorInfo& inputTensorInfo  = GetTensorInfoForTfLiteOpaqueTensor(tfLiteInputTensor);
    const armnn::TensorInfo& outputTensorInfo = GetTensorInfoForTfLiteOpaqueTensor(tfLiteOutputTensor, true);

    bool             isSupported  = false;
    armnn::BackendId setBackend;
    auto             validateFunc = [&](const armnn::TensorInfo& outInfo, bool& isSupported) {
        FORWARD_LAYER_OPAQUE_SUPPORT_FUNC("CAST",
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
    auto layerName = GetName(armnn::LayerType::Cast, nodeIndex);
    armnn::IConnectableLayer* layer = delegateData.m_Network->AddCastLayer(layerName.c_str());
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
    return Connect(layer, tfLiteContext, tfLiteNode, delegateData);
}

TfLiteStatus VisitReshapeOperator(DelegateData& delegateData,
                                  TfLiteOpaqueContext* tfLiteContext,
                                  TfLiteOpaqueNode* tfLiteNode,
                                  int nodeIndex,
                                  int32_t operatorCode)
{
    auto numInputs = TfLiteOpaqueNodeNumberOfInputs(tfLiteNode);

    if (numInputs == 2)
    {
        TF_LITE_ENSURE_STATUS(ValidateNumInputs(tfLiteContext, tfLiteNode, 2, nodeIndex));
    }
    else
    {
        TF_LITE_ENSURE_STATUS(ValidateNumInputs(tfLiteContext, tfLiteNode, 1, nodeIndex));
    }
    TF_LITE_ENSURE_STATUS(ValidateNumOutputs(tfLiteContext, tfLiteNode, 1, nodeIndex));

    // Gather input indices and use to get input tensor.
    const int* inputTensors;
    if (TfLiteOpaqueNodeInputs(tfLiteNode, &inputTensors, &numInputs) != kTfLiteOk)
    {
        TF_LITE_OPAQUE_MAYBE_KERNEL_LOG(
                tfLiteContext,
                "TfLiteArmnnOpaqueDelegate: Unable to gather input tensor indices from node #%d: ",
                nodeIndex);
        return kTfLiteError;
    }

    const TfLiteOpaqueTensor* tfLiteInputTensor = TfLiteOpaqueContextGetOpaqueTensor(tfLiteContext, inputTensors[0]);
    if (!IsValid(tfLiteContext, tfLiteInputTensor, operatorCode, nodeIndex))
    {
        return kTfLiteError;
    }

    // Gather output indices and use to get output tensors.
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

    const TfLiteOpaqueTensor* tfLiteOutputTensor = TfLiteOpaqueContextGetOpaqueTensor(tfLiteContext, outputTensors[0]);
    if (!IsValid(tfLiteContext, tfLiteOutputTensor, operatorCode, nodeIndex))
    {
        return kTfLiteError;
    }

    const armnn::TensorInfo& inputTensorInfo0 = GetTensorInfoForTfLiteOpaqueTensor(tfLiteInputTensor);
    const armnn::TensorInfo& outputTensorInfo = GetTensorInfoForTfLiteOpaqueTensor(tfLiteOutputTensor, true);

    // Check for unsupported 0-size dimensions in the input/output tensor shapes
    if(ZeroDimPresent({inputTensorInfo0, outputTensorInfo}))
    {
        TF_LITE_OPAQUE_MAYBE_KERNEL_LOG(
            tfLiteContext,
            "TfLiteArmnnOpaqueDelegate: Zero dimension tensors are not supported in operator #%d node #%d",
            operatorCode, nodeIndex);
        return kTfLiteError;
    }

    armnn::ReshapeDescriptor reshapeDesc;
    std::vector<int32_t> targetShape;

    auto* reshapeOptions = reinterpret_cast<TfLiteReshapeParams*>(TfLiteOpaqueNodeGetBuiltinData(tfLiteNode));

    // The new shape can be defined by either a second input tensor or by a builtin option, we need to check for both.
    // Options might be set without valid data. we need to check the dimensions are in a valid range.
    if (reshapeOptions && reshapeOptions->num_dimensions > 0 && reshapeOptions->num_dimensions <= 8)
    {
        for (int i = 0; i < reshapeOptions->num_dimensions; ++i)
        {
            targetShape.push_back(reshapeOptions->shape[i]);
        }
    }
    else if (numInputs == 2)
    {
        // Get shape from the second input tensor
        const TfLiteOpaqueTensor* tfLiteShapeInputTensor =
                TfLiteOpaqueContextGetOpaqueTensor(tfLiteContext, inputTensors[1]);
        if (!IsValid(tfLiteContext, tfLiteShapeInputTensor, operatorCode, nodeIndex))
        {
            return kTfLiteError;
        }

        int32_t numDims = TfLiteOpaqueTensorNumDims(tfLiteShapeInputTensor);
        if (numDims != 1)
        {
            TF_LITE_OPAQUE_MAYBE_KERNEL_LOG(
                    tfLiteContext,
                    "TfLiteArmnnOpaqueDelegate: Target 'shape' input is not a 1D tensor in "
                    "operator #%d node #%d: Falling back to TfLiteOptions.",
                    operatorCode, nodeIndex);
        }
        else
        {
            // Get the shape data out of the input tensor
            auto* shapeTensorDataPtr = static_cast<int32_t*>(TfLiteOpaqueTensorData(tfLiteShapeInputTensor));
            int32_t shapeTensorNumValues = TfLiteOpaqueTensorDim(tfLiteShapeInputTensor, 0);
            for (int32_t i = 0; i < shapeTensorNumValues; ++i)
            {
                targetShape.push_back(shapeTensorDataPtr[i]);
            }
        }
    }
    else
    {
        TF_LITE_OPAQUE_MAYBE_KERNEL_LOG(
                tfLiteContext,
                "TfLiteArmnnOpaqueDelegate: Target shape not defined in reshape parameters or input tensor. "
                "At least one method required in operator #%d node #%d: ",
                operatorCode, nodeIndex);
        return kTfLiteError;
    }

    // Check the target shape to check if there is zero in the shape.
    if (std::find(targetShape.begin(), targetShape.end(), 0) != targetShape.end() &&
        inputTensorInfo0.GetNumElements() != 0)
    {
        TF_LITE_OPAQUE_MAYBE_KERNEL_LOG(
                tfLiteContext,
                "TfLiteArmnnOpaqueDelegate: Input to reshape is a tensor with elements, "
                "but the requested shape has 0. "
                "operator #%d node #%d: ",
                operatorCode, nodeIndex);
        return kTfLiteError;
    }

    // Use the data to create the required tensor shape.
    if (CreateOutputTensorShape(inputTensorInfo0, targetShape, reshapeDesc) != kTfLiteOk)
    {
        TF_LITE_OPAQUE_MAYBE_KERNEL_LOG(
                tfLiteContext,
                "TfLiteArmnnOpaqueDelegate: At most one component of shape can be -1 in: "
                "operator #%d node #%d: ",
                operatorCode, nodeIndex);
        return kTfLiteError;
    }

    if (reshapeDesc.m_TargetShape.GetNumElements() != inputTensorInfo0.GetNumElements())
    {
        TF_LITE_OPAQUE_MAYBE_KERNEL_LOG(
                tfLiteContext,
                "TfLiteArmnnOpaqueDelegate: Reshape, number of elements in output shape does not match input "
                "operator #%d node #%d: ",
                operatorCode, nodeIndex);
        return kTfLiteError;
    }

    bool isSupported = false;
    armnn::BackendId setBackend;
    auto validateFunc = [&](const armnn::TensorInfo& outInfo, bool& isSupported)
    {
        FORWARD_LAYER_OPAQUE_SUPPORT_FUNC("RESHAPE",
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

    auto layerName = GetName(armnn::LayerType::Reshape, nodeIndex);
    armnn::IConnectableLayer* layer = delegateData.m_Network->AddReshapeLayer(reshapeDesc, layerName.c_str());
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
    return Connect(layer, tfLiteContext, tfLiteNode, delegateData);
}

TfLiteStatus VisitSqueezeOperator(DelegateData& delegateData,
                                  TfLiteOpaqueContext* tfLiteContext,
                                  TfLiteOpaqueNode* tfLiteNode,
                                  int nodeIndex,
                                  int32_t operatorCode)
{
    TF_LITE_ENSURE_STATUS(ValidateNumInputs(tfLiteContext, tfLiteNode, 1, nodeIndex));
    TF_LITE_ENSURE_STATUS(ValidateNumOutputs(tfLiteContext, tfLiteNode, 1, nodeIndex));

    // Gather input indices and use to get input tensor.
    int numInputs = 0;
    const int* inputTensors;
    if (TfLiteOpaqueNodeInputs(tfLiteNode, &inputTensors, &numInputs) != kTfLiteOk)
    {
        TF_LITE_OPAQUE_MAYBE_KERNEL_LOG(
                tfLiteContext,
                "TfLiteArmnnOpaqueDelegate: Unable to gather input tensor indices from node #%d: ",
                nodeIndex);
        return kTfLiteError;
    }

    const TfLiteOpaqueTensor* tfLiteInputTensor = TfLiteOpaqueContextGetOpaqueTensor(tfLiteContext, inputTensors[0]);
    if (!IsValid(tfLiteContext, tfLiteInputTensor, operatorCode, nodeIndex))
    {
        return kTfLiteError;
    }

    // Gather output indices and use to get output tensors.
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

    const TfLiteOpaqueTensor* tfLiteOutputTensor = TfLiteOpaqueContextGetOpaqueTensor(tfLiteContext, outputTensors[0]);
    if (!IsValid(tfLiteContext, tfLiteOutputTensor, operatorCode, nodeIndex))
    {
        return kTfLiteError;
    }

    auto* options = reinterpret_cast<TfLiteSqueezeParams*>(TfLiteOpaqueNodeGetBuiltinData(tfLiteNode));

    const armnn::TensorInfo& inputTensorInfo = GetTensorInfoForTfLiteOpaqueTensor(tfLiteInputTensor);

    std::vector<uint32_t> squeezeDim;
    // A single negative dim index is interpreted as a negative index in python
    // Meaning the index will be the shape size plus the negative index value
    if (options->num_squeeze_dims == 1 && options->squeeze_dims[0] < 0)
    {
        int32_t dim = static_cast<int32_t>(inputTensorInfo.GetShape().GetNumDimensions()) + options->squeeze_dims[0];
        squeezeDim.push_back(static_cast<uint32_t>(dim));
    }
    else
    {
        for (int32_t i = 0; i < options->num_squeeze_dims; ++i)
        {
            squeezeDim.push_back(static_cast<uint32_t>(options->squeeze_dims[i]));
        }
    }

    armnn::TensorInfo outputTensorInfo = OutputShapeOfSqueeze(squeezeDim, inputTensorInfo);

    armnn::ReshapeDescriptor reshapeDesc;
    reshapeDesc.m_TargetShape = outputTensorInfo.GetShape();

    bool isSupported = false;
    armnn::BackendId setBackend;
    auto validateFunc = [&](const armnn::TensorInfo& outInfo, bool& isSupported)
    {
        FORWARD_LAYER_OPAQUE_SUPPORT_FUNC("SQUEEZE",
                                          tfLiteContext,
                                          IsReshapeSupported,
                                          delegateData.m_Backends,
                                          isSupported,
                                          setBackend,
                                          inputTensorInfo,
                                          outInfo,
                                          reshapeDesc);
    };

    if (!delegateData.m_Network)
    {
        validateFunc(outputTensorInfo, isSupported);
        return isSupported ? kTfLiteOk : kTfLiteError;
    }

    auto layerName = GetName(armnn::LayerType::Reshape, nodeIndex, "Squeeze");
    armnn::IConnectableLayer* layer = delegateData.m_Network->AddReshapeLayer(reshapeDesc, layerName.c_str());
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
    return Connect(layer, tfLiteContext, tfLiteNode, delegateData);
}

TfLiteStatus VisitExpandDimsOperator(DelegateData& delegateData,
                                     TfLiteOpaqueContext* tfLiteContext,
                                     TfLiteOpaqueNode* tfLiteNode,
                                     int nodeIndex,
                                     int32_t operatorCode)
{
    TF_LITE_ENSURE_STATUS(ValidateNumInputs(tfLiteContext, tfLiteNode, 2, nodeIndex));
    TF_LITE_ENSURE_STATUS(ValidateNumOutputs(tfLiteContext, tfLiteNode, 1, nodeIndex));

    // Gather input indices and use to get input tensor.
    int numInputs = 0;
    const int* inputTensors;
    if (TfLiteOpaqueNodeInputs(tfLiteNode, &inputTensors, &numInputs) != kTfLiteOk)
    {
        TF_LITE_OPAQUE_MAYBE_KERNEL_LOG(
                tfLiteContext,
                "TfLiteArmnnOpaqueDelegate: Unable to gather input tensor indices from node #%d: ",
                nodeIndex);
        return kTfLiteError;
    }

    const TfLiteOpaqueTensor* tfLiteInputTensor = TfLiteOpaqueContextGetOpaqueTensor(tfLiteContext, inputTensors[0]);
    if (!IsValid(tfLiteContext, tfLiteInputTensor, operatorCode, nodeIndex))
    {
        return kTfLiteError;
    }

    const TfLiteOpaqueTensor* tfLiteAxisTensor = TfLiteOpaqueContextGetOpaqueTensor(tfLiteContext, inputTensors[1]);
    if (!IsValid(tfLiteContext, tfLiteAxisTensor, operatorCode, nodeIndex))
    {
        return kTfLiteError;
    }

    // Gather output indices and use to get output tensors.
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

    TfLiteOpaqueTensor* tfLiteOutputTensor = TfLiteOpaqueContextGetOpaqueTensor(tfLiteContext, outputTensors[0]);
    if (!IsValid(tfLiteContext, tfLiteOutputTensor, operatorCode, nodeIndex))
    {
        return kTfLiteError;
    }

    const armnn::TensorInfo& inputTensorInfo = GetTensorInfoForTfLiteOpaqueTensor(tfLiteInputTensor);
    armnn::TensorInfo outputTensorInfo = GetTensorInfoForTfLiteOpaqueTensor(tfLiteOutputTensor);

    auto* axisTensorData = static_cast<int32_t*>(TfLiteOpaqueTensorData(tfLiteAxisTensor));
    int32_t axis = axisTensorData[0];

    int32_t inputDimSize = static_cast<int32_t>(inputTensorInfo.GetShape().GetNumDimensions());
    if (axis > inputDimSize || axis < 0 - (inputDimSize + 1))
    {
        TF_LITE_OPAQUE_MAYBE_KERNEL_LOG(
                tfLiteContext,
                "TfLiteArmnnOpaqueDelegate: Axis must be in range "
                "[0 - (inputDimSize + 1), inputDimSize] inclusive.");
        return kTfLiteError;
    }

    if(axis < 0)
    {
        axis = inputDimSize + axis + 1;
    }

    std::vector<unsigned int> shape(static_cast<unsigned int>(inputDimSize) + 1);
    unsigned int inputShapeIndex = 0;
    for (unsigned int i = 0; i < static_cast<unsigned int>(inputDimSize + 1); ++i)
    {
        if (i == static_cast<unsigned int>(axis))
        {
            shape[i] = 1;
        }
        else
        {
            shape[i] = inputTensorInfo.GetShape()[inputShapeIndex];
            ++inputShapeIndex;
        }
    }

    armnn::ReshapeDescriptor reshapeDesc;
    reshapeDesc.m_TargetShape = armnn::TensorShape(static_cast<unsigned int>(inputDimSize + 1), shape.data());

    bool isSupported = false;
    armnn::BackendId setBackend;
    auto validateFunc = [&](const armnn::TensorInfo& outInfo, bool& isSupported)
    {
        FORWARD_LAYER_OPAQUE_SUPPORT_FUNC("EXPAND_DIMS",
                                          tfLiteContext,
                                          IsReshapeSupported,
                                          delegateData.m_Backends,
                                          isSupported,
                                          setBackend,
                                          inputTensorInfo,
                                          outInfo,
                                          reshapeDesc);
    };

    if (!delegateData.m_Network)
    {
        validateFunc(outputTensorInfo, isSupported);
        return isSupported ? kTfLiteOk : kTfLiteError;
    }

    auto layerName = GetName(armnn::LayerType::Reshape, nodeIndex, "ExpandDims");
    armnn::IConnectableLayer* layer = delegateData.m_Network->AddReshapeLayer(reshapeDesc, layerName.c_str());
    layer->SetBackendId(setBackend);
    ARMNN_ASSERT(layer != nullptr);

    armnn::IOutputSlot& outputSlot = layer->GetOutputSlot(0);
    outputTensorInfo.SetShape(reshapeDesc.m_TargetShape);
    outputSlot.SetTensorInfo(outputTensorInfo);

    // try to connect the Constant Inputs if there are any
    if (ProcessInputs(layer, delegateData, tfLiteContext, tfLiteNode, nodeIndex) != kTfLiteOk)
    {
        return kTfLiteError;
    }

    // Connect
    return Connect(layer, tfLiteContext, tfLiteNode, delegateData);
}

}
