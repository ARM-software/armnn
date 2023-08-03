//
// Copyright © 2023 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include <OpaqueDelegateUtils.hpp>

namespace armnnOpaqueDelegate
{

TfLiteStatus ValidateResizeOperator(DelegateData& delegateData,
                                    TfLiteOpaqueContext* tfLiteContext,
                                    const armnn::TensorInfo& inputInfo,
                                    const armnn::TensorInfo& outputInfo,
                                    const armnn::ResizeDescriptor& descriptor)
{
    bool isSupported = false;
    FORWARD_LAYER_OPAQUE_SUPPORT_FUNC("RESIZE",
                                       tfLiteContext,
                                       IsResizeSupported,
                                       delegateData.m_Backends,
                                       isSupported,
                                       armnn::BackendId(),
                                       inputInfo,
                                       outputInfo,
                                       descriptor);

    return isSupported ? kTfLiteOk : kTfLiteError;
}

TfLiteStatus VisitResizeOperator(DelegateData& delegateData,
                                 TfLiteOpaqueContext* tfLiteContext,
                                 TfLiteOpaqueNode* tfLiteNode,
                                 int nodeIndex,
                                 int32_t resizeOperatorCode)
{
    TF_LITE_ENSURE_STATUS(ValidateNumInputs(tfLiteContext, tfLiteNode, 2, nodeIndex));
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

    // The first input contains the data of the image that should be resized [batch, height, width, channels]
    const TfLiteOpaqueTensor* tfLiteInputTensor =
            TfLiteOpaqueContextGetOpaqueTensor(tfLiteContext, inputTensors[0]);
    if (IsDynamicTensor(tfLiteInputTensor))
    {
        TF_LITE_OPAQUE_MAYBE_KERNEL_LOG(
                tfLiteContext,
                "TfLiteArmnnOpaqueDelegate: Dynamic input tensors are not supported in operator #%d node #%d: ",
                resizeOperatorCode, nodeIndex);
        return kTfLiteError;
    }

    // The second input contains a size tensor. The size tensor contains two integer values
    // that describe the new height and width of the image [new_height, new_width]
    const TfLiteOpaqueTensor* tfLiteSizeTensor =
            TfLiteOpaqueContextGetOpaqueTensor(tfLiteContext, inputTensors[1]);
    if (IsDynamicTensor(tfLiteSizeTensor))
    {
        TF_LITE_OPAQUE_MAYBE_KERNEL_LOG(
                tfLiteContext,
                "TfLiteArmnnOpaqueDelegate: Dynamic input tensors are not supported in operator #%d node #%d: ",
                resizeOperatorCode, nodeIndex);
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

    // The output tensor should have the shape [batch, new_height, new_width, channels]
    const TfLiteOpaqueTensor* tfLiteOutputTensor =
            TfLiteOpaqueContextGetOpaqueTensor(tfLiteContext, outputTensors[0]);
    if (IsDynamicTensor(tfLiteOutputTensor))
    {
        TF_LITE_OPAQUE_MAYBE_KERNEL_LOG(
                tfLiteContext,
                "TfLiteArmnnOpaqueDelegate: Dynamic output tensors are not supported in operator #%d node #%d: ",
                resizeOperatorCode, nodeIndex);
        return kTfLiteError;
    }

    const armnn::TensorInfo& inputTensorInfo =
            GetTensorInfoForTfLiteOpaqueTensor(tfLiteInputTensor);
    const armnn::TensorInfo& outputTensorInfo =
            GetTensorInfoForTfLiteOpaqueTensor(tfLiteOutputTensor, true);

    std::string layerName("Resize");

    // Fill descriptor
    armnn::ResizeDescriptor desc;
    switch (resizeOperatorCode)
    {
        case kTfLiteBuiltinResizeBilinear:
        {
            desc.m_Method = armnn::ResizeMethod::Bilinear;

            layerName += "Bilinear:" + std::to_string(nodeIndex);

            TfLiteResizeBilinearParams* bilinearOptions =
                    reinterpret_cast<TfLiteResizeBilinearParams*>(TfLiteOpaqueNodeGetBuiltinData(tfLiteNode));

            desc.m_AlignCorners = bilinearOptions->align_corners;
            desc.m_HalfPixelCenters = bilinearOptions->half_pixel_centers;
            break;
        }
        case kTfLiteBuiltinResizeNearestNeighbor:
        {
            desc.m_Method =  armnn::ResizeMethod::NearestNeighbor;
            layerName += "NearestNeighbor:" + std::to_string(nodeIndex);

            TfLiteResizeNearestNeighborParams* nearestNeighborOptions =
                    reinterpret_cast<TfLiteResizeNearestNeighborParams*>(TfLiteOpaqueNodeGetBuiltinData(tfLiteNode));

            desc.m_AlignCorners = nearestNeighborOptions->align_corners;
            desc.m_HalfPixelCenters = nearestNeighborOptions->half_pixel_centers;
            break;
        }
        default:
        {
            TF_LITE_OPAQUE_MAYBE_KERNEL_LOG(
                    tfLiteContext,
                    "TfLiteArmnnOpaqueDelegate: Unknown TfLite built in operation for Resize. "
                    "Given operator: #%d node #%d: ",
                    resizeOperatorCode, nodeIndex);
            return kTfLiteError;
        }
    }

    // In Arm NN the values of the size input tensor [new_height, new_width] is saved in the operator
    // descriptor. We have to read it from the input tensor and write it to the descriptor.

    auto* sizeTensorDataPtr = static_cast<int*>(TfLiteOpaqueTensorData(tfLiteSizeTensor));
    auto sizeTensorNumDimensions = TfLiteOpaqueTensorNumDims(tfLiteSizeTensor);
    // The size tensor is only a 1D tensor -> [new_height, new width]
    if (sizeTensorNumDimensions != 1)
    {
        TF_LITE_OPAQUE_MAYBE_KERNEL_LOG(
                tfLiteContext,
                "TfLiteArmnnOpaqueDelegate: The Size-Input-Tensor of the Resize operation is not allowed to be a "
                "dynamic tensor. Operator: #%d node #%d: ",
                resizeOperatorCode, nodeIndex);
        return kTfLiteError;
    }

    // Get number of values in the size tensor
    auto sizeTensorNumValues = TfLiteOpaqueTensorDim(tfLiteSizeTensor,0);
    if (sizeTensorNumValues == 0)
    {
        TF_LITE_OPAQUE_MAYBE_KERNEL_LOG(
                tfLiteContext,
                "TfLiteArmnnOpaqueDelegate: The Size-Input-Tensor of the Resize operation is not allowed to be a "
                "dynamic tensor. Operator: #%d node #%d: ",
                resizeOperatorCode, nodeIndex);
        return kTfLiteError;
    }
    else if (sizeTensorNumValues != 2)
    {
        TF_LITE_OPAQUE_MAYBE_KERNEL_LOG(
                tfLiteContext,
                "TfLiteArmnnOpaqueDelegate: The Size-Input-Tensor of the Resize operation requires to "
                "have a dimension of 2 [new_height, new width] but a tensor with a dimension of #%d was given. "
                "Operator: #%d node #%d: ",
                sizeTensorNumValues, resizeOperatorCode, nodeIndex);
        return kTfLiteError;
    }
    // get size tensor data
    std::vector<int32_t> sizeTensorData(sizeTensorDataPtr, sizeTensorDataPtr+sizeTensorNumValues);

    desc.m_TargetHeight = static_cast<uint32_t> (sizeTensorData[0]);
    desc.m_TargetWidth  = static_cast<uint32_t> (sizeTensorData[1]);
    desc.m_DataLayout   = armnn::DataLayout::NHWC;

    // No network pointer indicates that only support for this operator should be checked
    if (!delegateData.m_Network)
    {
        return ValidateResizeOperator(delegateData,
                                      tfLiteContext,
                                      inputTensorInfo,
                                      outputTensorInfo,
                                      desc);
    }


    armnn::IConnectableLayer* resizeLayer = nullptr;
    layerName += ":";
    layerName += nodeIndex;

    resizeLayer = delegateData.m_Network->AddResizeLayer(desc, layerName.c_str());

    armnn::IOutputSlot& outputSlot = resizeLayer->GetOutputSlot(0);
    outputSlot.SetTensorInfo(outputTensorInfo);

    // try to connect the Constant Inputs if there are any
    if (ProcessInputs(resizeLayer, delegateData, tfLiteContext, tfLiteNode, nodeIndex) != kTfLiteOk)
    {
        return kTfLiteError;
    }

    ARMNN_ASSERT(resizeLayer != nullptr);

    return Connect(resizeLayer, tfLiteContext, tfLiteNode, delegateData);
}

} // namespace armnnOpaqueDelegate
