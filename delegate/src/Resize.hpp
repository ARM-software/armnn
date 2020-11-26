//
// Copyright © 2020 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include "DelegateUtils.hpp"
#include <armnn/utility/IgnoreUnused.hpp>

#include <armnn/Descriptors.hpp>

#include <tensorflow/lite/builtin_ops.h>
#include <tensorflow/lite/c/builtin_op_data.h>
#include <tensorflow/lite/c/common.h>
#include <tensorflow/lite/minimal_logging.h>
#include <tensorflow/lite/kernels/internal/tensor_ctypes.h>

namespace armnnDelegate
{



TfLiteStatus ValidateResizeOperator(DelegateData& delegateData,
                                    TfLiteContext* tfLiteContext,
                                    const armnn::TensorInfo& inputInfo,
                                    const armnn::TensorInfo& outputInfo,
                                    const armnn::ResizeDescriptor& descriptor)
{
    bool isSupported = false;
    FORWARD_LAYER_SUPPORT_FUNC(__func__,
                               tfLiteContext,
                               IsResizeSupported,
                               delegateData.m_Backends,
                               isSupported,
                               inputInfo,
                               outputInfo,
                               descriptor);

    return isSupported ? kTfLiteOk : kTfLiteError;
}

TfLiteStatus VisitResizeOperator(DelegateData& delegateData,
                                 TfLiteContext* tfLiteContext,
                                 TfLiteNode* tfLiteNode,
                                 int nodeIndex,
                                 int32_t resizeOperatorCode)
{
    TF_LITE_ENSURE_STATUS(ValidateNumInputs(tfLiteContext, tfLiteNode, 2, nodeIndex));
    TF_LITE_ENSURE_STATUS(ValidateNumOutputs(tfLiteContext, tfLiteNode, 1, nodeIndex));

    const TfLiteTensor* tfLiteTensors = tfLiteContext->tensors;

    // The first input contains the data of the image that should be resized [batch, height, width, channels]
    const TfLiteTensor& tfLiteInputTensor = tfLiteTensors[tfLiteNode->inputs->data[0]];
    if (IsDynamicTensor(tfLiteInputTensor))
    {
        TF_LITE_MAYBE_KERNEL_LOG(
            tfLiteContext,
            "TfLiteArmnnDelegate: Dynamic input tensors are not supported in operator #%d node #%d: ",
            resizeOperatorCode, nodeIndex);
        return kTfLiteError;
    }

    // The second input contains a size tensor. The size tensor contains two integer values
    // that describe the new height and width of the image [new_height, new_width]
    const TfLiteTensor& tfLiteSizeTensor = tfLiteTensors[tfLiteNode->inputs->data[1]];
    if (IsDynamicTensor(tfLiteSizeTensor))
    {
        TF_LITE_MAYBE_KERNEL_LOG(
            tfLiteContext,
            "TfLiteArmnnDelegate: Dynamic input tensors are not supported in operator #%d node #%d: ",
            resizeOperatorCode, nodeIndex);
        return kTfLiteError;
    }

    // The output tensor should have the shape [batch, new_height, new_width, channels]
    const TfLiteTensor& tfLiteOutputTensor = tfLiteTensors[tfLiteNode->outputs->data[0]];
    if (IsDynamicTensor(tfLiteOutputTensor))
    {
        TF_LITE_MAYBE_KERNEL_LOG(
            tfLiteContext,
            "TfLiteArmnnDelegate: Dynamic output tensors are not supported in operator #%d node #%d: ",
            resizeOperatorCode, nodeIndex);
        return kTfLiteError;
    }

    const armnn::TensorInfo& inputTensorInfo = GetTensorInfoForTfLiteTensor(tfLiteInputTensor);
    const armnn::TensorInfo& outputTensorInfo = GetTensorInfoForTfLiteTensor(tfLiteOutputTensor);

    std::string layerName("Resize");

    // Fill descriptor
    armnn::ResizeDescriptor desc;
    switch (resizeOperatorCode)
    {
        case kTfLiteBuiltinResizeBilinear:
        {
            desc.m_Method = armnn::ResizeMethod::Bilinear;

            layerName += "Bilinear:" + std::to_string(nodeIndex);

            TfLiteResizeBilinearParams* biliniarOptions =
                    reinterpret_cast<TfLiteResizeBilinearParams*>(tfLiteNode->builtin_data);

            desc.m_AlignCorners = biliniarOptions->align_corners;
            desc.m_HalfPixelCenters = biliniarOptions->half_pixel_centers;
            break;
        }
        case kTfLiteBuiltinResizeNearestNeighbor:
        {
            desc.m_Method =  armnn::ResizeMethod::NearestNeighbor;
            layerName += "NearestNeighbor:" + std::to_string(nodeIndex);

            TfLiteResizeNearestNeighborParams* nearestNeighborOptions =
                    reinterpret_cast<TfLiteResizeNearestNeighborParams*>(tfLiteNode->builtin_data);

            desc.m_AlignCorners = nearestNeighborOptions->align_corners;
            desc.m_HalfPixelCenters = nearestNeighborOptions->half_pixel_centers;
            break;
        }
        default:
        {
            TF_LITE_MAYBE_KERNEL_LOG(
                    tfLiteContext,
                    "TfLiteArmnnDelegate: Unknown TfLite built in operation for Resize. Given operator: #%d node #%d: ",
                    resizeOperatorCode, nodeIndex);
            return kTfLiteError;
        }
    }

    // In armnn the values of the size input tensor [new_hight, new_width] is saved in the operator
    // descriptor. We have to read it from the input tensor and write it to the descriptor.

    auto* sizeTensorDataPtr = tflite::GetTensorData<int32_t>(&tfLiteSizeTensor);
    auto sizeTensorNumDimensions = tfLiteSizeTensor.dims->size;
    // The size tensor is only a 1D tensor -> [new_hight, new width]
    if (sizeTensorNumDimensions != 1)
    {
        TF_LITE_MAYBE_KERNEL_LOG(
                tfLiteContext,
                "TfLiteArmnnDelegate: The Size-Input-Tensor of the Resize operation is not allowed to be a "
                "dynamic tensor. Operator: #%d node #%d: ",
                resizeOperatorCode, nodeIndex);
        return kTfLiteError;
    }

    // Get number of values in the size tensor
    auto sizeTensorNumValues = tfLiteSizeTensor.dims->data[0];
    if (sizeTensorNumValues == 0)
    {
        TF_LITE_MAYBE_KERNEL_LOG(
                tfLiteContext,
                "TfLiteArmnnDelegate: The Size-Input-Tensor of the Resize operation is not allowed to be a "
                "dynamic tensor. Operator: #%d node #%d: ",
                resizeOperatorCode, nodeIndex);
        return kTfLiteError;
    }
    else if (sizeTensorNumValues != 2)
    {
        TF_LITE_MAYBE_KERNEL_LOG(
                tfLiteContext,
                "TfLiteArmnnDelegate: The Size-Input-Tensor of the Resize operation requires to "
                "have a dimension of 2 [new_hight, new width] but a tensor with a dimension of #%d was given. "
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
    resizeLayer = delegateData.m_Network->AddResizeLayer(desc, layerName.c_str());

    armnn::IOutputSlot& outputSlot = resizeLayer->GetOutputSlot(0);
    outputSlot.SetTensorInfo(outputTensorInfo);

    ARMNN_ASSERT(resizeLayer != nullptr);

    return Connect(resizeLayer, tfLiteNode, delegateData);
}

} // namespace armnnDelegate
