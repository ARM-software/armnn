//
// Copyright © 2022-2024 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include <armnn/utility/IgnoreUnused.hpp>

#include <tensorflow/lite/builtin_ops.h>
#include <tensorflow/lite/c/builtin_op_data.h>
#include <tensorflow/lite/c/common.h>
#include <tensorflow/lite/minimal_logging.h>

namespace armnnDelegate
{

TfLiteStatus VisitStridedSliceOperator(DelegateData& delegateData,
                                       TfLiteContext* tfLiteContext,
                                       TfLiteNode* tfLiteNode,
                                       int nodeIndex,
                                       int32_t sliceOperatorCode)
{
    TF_LITE_ENSURE_STATUS(ValidateNumInputs(tfLiteContext, tfLiteNode, 4, nodeIndex));
    TF_LITE_ENSURE_STATUS(ValidateNumOutputs(tfLiteContext, tfLiteNode, 1, nodeIndex));

    // Read inputs [input, begin, end, strides]
    int numInputs = tfLiteNode->inputs->size;
    std::vector<const TfLiteTensor*> tfLiteInputs;
    tfLiteInputs.reserve(numInputs);
    const TfLiteTensor* tfLiteTensors = tfLiteContext->tensors;
    for (int i = 0; i < numInputs; i++)
    {
        const TfLiteTensor* inputTensor = &tfLiteTensors[tfLiteNode->inputs->data[i]];
        tfLiteInputs.push_back(inputTensor);
        if (!IsValid(tfLiteContext, *inputTensor, sliceOperatorCode, nodeIndex))
        {
            return kTfLiteError;
        }
        // Checking for unsupported non-const non-network input tensors
        // Index 0 is the input, index 1-3 should be constant
        if(i > 0 && inputTensor->allocation_type != kTfLiteMmapRo)
        {
            TF_LITE_MAYBE_KERNEL_LOG(
                tfLiteContext,
                "TfLiteArmnnDelegate: Unsupported constant data input through non-const tensor "
                "in operator #%d node #%d",
                sliceOperatorCode, nodeIndex);
            return kTfLiteError;
        }
    }

    // We save the begin, end and strides tensors in our descriptor. Therefore we have to read those values from inputs
    int inputRank = tfLiteInputs[0]->dims->size;

    // Input tensors of rank greater than 4 are unsupported - delegate back to TFLite runtime
    if(inputRank > 4)
    {
        TF_LITE_MAYBE_KERNEL_LOG(
            tfLiteContext,
            "TfLitearmnnOpaqueDelegate: Tensors of rank greater than 4 are unsupported"
            " in the StridedSlice operator. Operator: #%d node #%d: ",
            sliceOperatorCode, nodeIndex);
        return kTfLiteError;
    }

    auto ReadInt32Input = [&](int inputIndex, std::vector<int32_t>& outputData) ->  TfLiteStatus
    {
        if (tfLiteInputs[inputIndex]->type != kTfLiteInt32)
        {
            TF_LITE_MAYBE_KERNEL_LOG(
                tfLiteContext,
                "TfLiteArmnnDelegate: The Begin-, End- and Stride-Tensors of the StridedSlice operation need to "
                "be of type int32. Operator: #%d node #%d: ",
                sliceOperatorCode, nodeIndex);
            return kTfLiteError;
        }
        int rank = tfLiteInputs[inputIndex]->dims->size;
        if (rank != 1)
        {
            TF_LITE_MAYBE_KERNEL_LOG(
                tfLiteContext,
                "TfLiteArmnnDelegate: The Begin-, End- and Stride-Tensors of the StridedSlice operation need to "
                "be a 1D-Tensor. Operator: #%d node #%d: ",
                sliceOperatorCode, nodeIndex);
            return kTfLiteError;
        }
        int numValues = tfLiteInputs[inputIndex]->dims->data[0];
        if (numValues != inputRank)
        {
            TF_LITE_MAYBE_KERNEL_LOG(
                tfLiteContext,
                "TfLiteArmnnDelegate: The number of values in the Begin-, End- and Stride-Tensors of the "
                "StridedSlice operation need to be equal to the rank of the Input-Tensor. Operator: #%d node #%d: ",
                sliceOperatorCode, nodeIndex);
            return kTfLiteError;
        }
        // return tensor data
        auto* tensorDataPtr = tflite::GetTensorData<int32_t>(tfLiteInputs[inputIndex]);
        outputData.assign(tensorDataPtr, tensorDataPtr+numValues);
        return kTfLiteOk;
    };

    std::vector<int32_t> beginData;
    if (ReadInt32Input(1, beginData) != kTfLiteOk)
        return kTfLiteError;
    std::vector<int32_t> endData;
    if (ReadInt32Input(2, endData) != kTfLiteOk)
        return kTfLiteError;
    std::vector<int32_t> strideData;
    if (ReadInt32Input(3, strideData) != kTfLiteOk)
        return kTfLiteError;

    // parse built in options
    auto* stridedSliceParams = reinterpret_cast<TfLiteStridedSliceParams*>(tfLiteNode->builtin_data);

    // Write all data to the descriptor
    armnn::StridedSliceDescriptor descriptor;
    descriptor.m_Begin          = std::move(beginData);
    descriptor.m_End            = std::move(endData);
    descriptor.m_Stride         = std::move(strideData);
    descriptor.m_BeginMask      = stridedSliceParams->begin_mask;
    descriptor.m_EllipsisMask   = stridedSliceParams->ellipsis_mask;
    descriptor.m_EndMask        = stridedSliceParams->end_mask;
    descriptor.m_NewAxisMask    = stridedSliceParams->new_axis_mask;
    descriptor.m_ShrinkAxisMask = stridedSliceParams->shrink_axis_mask;
    descriptor.m_DataLayout     = armnn::DataLayout::NHWC;

    // Checking begin and end bounds with ShrinkAxisMask
    for(unsigned int i = 0; i < inputRank; ++i)
    {
        if((descriptor.m_ShrinkAxisMask & (1 << i)) &&
           (((descriptor.m_Begin[i] - descriptor.m_End[i]) > 1) ||
           ((descriptor.m_Begin[i] - descriptor.m_End[i]) < -1)))
        {
            TF_LITE_MAYBE_KERNEL_LOG(
                tfLiteContext,
                "TfLitearmnnDelegate: Invalid combination of ShrinkAxisMask, Begin- and End-Tensor values "
                "in the StridedSlice operator. Operator: #%d node #%d: ",
                sliceOperatorCode, nodeIndex);
            return kTfLiteError;
        }
    }

    // Checking that NewAxisMask doesn't extend the output beyond the supported rank
    if(inputRank >= 3 && (descriptor.m_NewAxisMask > 4 || descriptor.m_NewAxisMask == 3))
    {
        TF_LITE_MAYBE_KERNEL_LOG(
            tfLiteContext,
            "TfLitearmnnOpaqueDelegate: Maximum output tensor rank is 4, the currently set NewAxisMask "
            "results in an unsupported higher rank. Operator: #%d node #%d: ",
            sliceOperatorCode, nodeIndex);
        return kTfLiteError;
    }

    // The variable 'offset' is documented in TFLite builtin_op_data.h:
    // "If true, then the end tensor is an offset of the begin tensor."
    if(stridedSliceParams->offset &&
       descriptor.m_Begin.size() == descriptor.m_End.size())
    {
        for(unsigned int i = 0; i < descriptor.m_End.size(); ++i)
        {
            descriptor.m_End[i] += descriptor.m_Begin[i];
        }
    }

    // Validate output
    const TfLiteTensor& tfLiteOutputTensor = tfLiteTensors[tfLiteNode->outputs->data[0]];
    if (!IsValid(tfLiteContext, tfLiteOutputTensor, sliceOperatorCode, nodeIndex))
    {
        return kTfLiteError;
    }

    const armnn::TensorInfo& inputTensorInfo  = GetTensorInfoForTfLiteTensor(*tfLiteInputs[0]);
    const armnn::TensorInfo& outputTensorInfo = GetTensorInfoForTfLiteTensor(tfLiteOutputTensor);

    // Check for unsupported 0-size dimensions in the input/output tensor shapes
    if(ZeroDimPresent({inputTensorInfo, outputTensorInfo}))
    {
        TF_LITE_MAYBE_KERNEL_LOG(
            tfLiteContext,
            "TfLiteArmnnOpaqueDelegate: Zero dimension tensors are not supported in operator #%d node #%d",
            sliceOperatorCode, nodeIndex);
        return kTfLiteError;
    }

    bool isSupported = false;
    armnn::BackendId setBackend;
    auto validateFunc = [&](const armnn::TensorInfo& outInfo, bool& isSupported)
    {
        FORWARD_LAYER_SUPPORT_FUNC("STRIDED_SLICE",
                                   tfLiteContext,
                                   IsStridedSliceSupported,
                                   delegateData.m_Backends,
                                   isSupported,
                                   setBackend,
                                   inputTensorInfo,
                                   outInfo,
                                   descriptor);
    };

    if (!delegateData.m_Network)
    {
        validateFunc(outputTensorInfo, isSupported);
        return isSupported ? kTfLiteOk : kTfLiteError;
    }

    // Add a StridedSlice layer
    auto layerName = GetLayerName(armnn::LayerType::StridedSlice, nodeIndex);
    armnn::IConnectableLayer* layer = delegateData.m_Network->AddStridedSliceLayer(descriptor, layerName.c_str());
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
    return Connect(layer, tfLiteNode, delegateData);
}

} // namespace armnnDelegate

