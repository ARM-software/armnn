//
// Copyright © 2020 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include "DelegateUtils.hpp"

#include <tensorflow/lite/builtin_ops.h>
#include <tensorflow/lite/c/builtin_op_data.h>
#include <tensorflow/lite/c/common.h>
#include <tensorflow/lite/minimal_logging.h>
#include "tensorflow/lite/kernels/internal/tensor.h"

namespace armnnDelegate
{

TfLiteStatus VisitConv2dOperator(DelegateData& delegateData,
                                 TfLiteContext* tfLiteContext,
                                 TfLiteNode* tfLiteNode,
                                 int nodeIndex,
                                 int32_t operatorCode)
{
    auto numInputs = tfLiteNode->inputs->size;
    if (numInputs < 2)
    {
        TF_LITE_MAYBE_KERNEL_LOG(
            tfLiteContext, "TfLiteArmnnDelegate: Minimum number of inputs (%d != %d) in node #%d",
            2, numInputs, nodeIndex);
        return kTfLiteError;
    }
    TF_LITE_ENSURE_STATUS(ValidateNumOutputs(tfLiteContext, tfLiteNode, 1, nodeIndex));

    armnn::Convolution2dDescriptor descriptor;
    const auto params = reinterpret_cast<TfLiteConvParams*>(tfLiteNode->builtin_data);

    bool biasEnabled = tfLiteNode->inputs->size > 2;
    descriptor.m_BiasEnabled = biasEnabled;
    descriptor.m_StrideX = NonNegative(params->stride_width, nodeIndex);
    descriptor.m_StrideY = NonNegative(params->stride_height, nodeIndex);
    descriptor.m_DataLayout = armnn::DataLayout::NHWC;
    descriptor.m_DilationX = NonNegative(params->dilation_width_factor, nodeIndex);
    descriptor.m_DilationY = NonNegative(params->dilation_height_factor, nodeIndex);

    const TfLiteTensor* tfLiteTensors = tfLiteContext->tensors;
    const TfLiteTensor& tfLiteInputTensor = tfLiteTensors[tfLiteNode->inputs->data[0]];
    if(!IsValid(&tfLiteTensors[tfLiteNode->inputs->data[0]]))
    {
        TF_LITE_MAYBE_KERNEL_LOG(
            tfLiteContext,
            "TfLiteArmnnDelegate: Invalid input tensor in operator #%d node #%d: ",
            operatorCode, nodeIndex);
        return kTfLiteError;
    }
    if (IsDynamicTensor(tfLiteInputTensor))
    {
        TF_LITE_MAYBE_KERNEL_LOG(
            tfLiteContext,
            "TfLiteArmnnDelegate: Dynamic input tensors are not supported in operator #%d node #%d: ",
            operatorCode, nodeIndex);
        return kTfLiteError;
    }
    const TfLiteTensor& tfLiteOutputTensor = tfLiteTensors[tfLiteNode->outputs->data[0]];
    if(!IsValid(&tfLiteOutputTensor))
    {
        TF_LITE_MAYBE_KERNEL_LOG(
            tfLiteContext,
            "TfLiteArmnnDelegate: Invalid output tensor in operator #%d node #%d: ",
            operatorCode, nodeIndex);
        return kTfLiteError;
    }
    if (IsDynamicTensor(tfLiteOutputTensor))
    {
        TF_LITE_MAYBE_KERNEL_LOG(
            tfLiteContext,
            "TfLiteArmnnDelegate: Dynamic output tensors are not supported in operator #%d node #%d: ",
            operatorCode, nodeIndex);
        return kTfLiteError;
    }

    const TfLiteTensor& tfLiteFilterTensor = tfLiteTensors[tfLiteNode->inputs->data[1]];
    if(!IsValid(&tfLiteFilterTensor))
    {
        TF_LITE_MAYBE_KERNEL_LOG(
            tfLiteContext,
            "TfLiteArmnnDelegate: Invalid filter tensor in operator #%d node #%d: ",
            operatorCode, nodeIndex);
        return kTfLiteError;
    }
    if (IsDynamicTensor(tfLiteFilterTensor))
    {
        TF_LITE_MAYBE_KERNEL_LOG(
            tfLiteContext,
            "TfLiteArmnnDelegate: Dynamic filter tensors are not supported in node #%d: ",
            nodeIndex);
        return kTfLiteError;
    }

    const armnn::TensorInfo& inputTensorInfo  = GetTensorInfoForTfLiteTensor(tfLiteInputTensor);
    const armnn::TensorInfo& outputTensorInfo = GetTensorInfoForTfLiteTensor(tfLiteOutputTensor);

    armnn::TensorInfo filterTensorInfo = GetTensorInfoForTfLiteTensor(tfLiteFilterTensor);

    armnn::TensorInfo biasTensorInfo;
    if(biasEnabled)
    {
        const TfLiteTensor& tfLiteBiasTensor = tfLiteTensors[tfLiteNode->inputs->data[2]];
        if(!IsValid(&tfLiteBiasTensor))
        {
            TF_LITE_MAYBE_KERNEL_LOG(
                tfLiteContext,
                "TfLiteArmnnDelegate: Invalid bias tensor in operator #%d node #%d: ",
                operatorCode, nodeIndex);
            return kTfLiteError;
        }
        if (IsDynamicTensor(tfLiteBiasTensor))
        {
            TF_LITE_MAYBE_KERNEL_LOG(
                tfLiteContext,
                "TfLiteArmnnDelegate: Dynamic bias tensors are not supported in node #%d: ",
                nodeIndex);
            return kTfLiteError;
        }
        biasTensorInfo = GetTensorInfoForTfLiteTensor(tfLiteBiasTensor);
    }
    else
    {
        biasTensorInfo = armnn::TensorInfo(armnn::TensorShape({1}), GetDataType(tfLiteInputTensor));
    }

    armnn::Optional<armnn::TensorInfo> optionalBiasInfo(biasTensorInfo);

    // TfLite uses NHWC tensors
    const unsigned int inputHeight = inputTensorInfo.GetShape()[1];
    const unsigned int inputWidth  = inputTensorInfo.GetShape()[2];

    const unsigned int filterHeight = filterTensorInfo.GetShape()[1];
    const unsigned int filterWidth  = filterTensorInfo.GetShape()[2];

    // Calculate padding
    CalcPadding(inputHeight, filterHeight, descriptor.m_StrideY, descriptor.m_DilationY,
                descriptor.m_PadTop, descriptor.m_PadBottom, params->padding);
    CalcPadding(inputWidth, filterWidth, descriptor.m_StrideX, descriptor.m_DilationX,
                descriptor.m_PadLeft, descriptor.m_PadRight, params->padding);

    if (!delegateData.m_Network)
    {
        bool isSupported = false;
        FORWARD_LAYER_SUPPORT_FUNC(__func__,
                                   tfLiteContext,
                                   IsConvolution2dSupported,
                                   delegateData.m_Backends,
                                   isSupported,
                                   inputTensorInfo,
                                   outputTensorInfo,
                                   descriptor,
                                   filterTensorInfo,
                                   optionalBiasInfo);
        return isSupported ? kTfLiteOk : kTfLiteError;
    }

    armnn::IConnectableLayer* layer = nullptr;

    // Set up filter and biases
    auto filter =
        CreateConstTensor(&tfLiteContext->tensors[tfLiteNode->inputs->data[1]],
                          filterTensorInfo,
                          armnn::Optional<armnn::PermutationVector&>());

    if(biasEnabled)
    {
        auto biases =
            CreateConstTensor(&tfLiteContext->tensors[tfLiteNode->inputs->data[2]],
                              biasTensorInfo,
                              armnn::Optional<armnn::PermutationVector&>());
        layer = delegateData.m_Network->AddConvolution2dLayer(descriptor,
                                                              filter,
                                                              armnn::Optional<armnn::ConstTensor>(biases));
    }
    else
    {
        layer = delegateData.m_Network->AddConvolution2dLayer(descriptor,
                                                              filter,
                                                              armnn::EmptyOptional());
    }

    ARMNN_ASSERT(layer != nullptr);

    armnn::IOutputSlot& outputSlot = layer->GetOutputSlot(0);
    outputSlot.SetTensorInfo(outputTensorInfo);

    Connect(layer, tfLiteNode, delegateData);

    auto* tfLiteNodeParameters = reinterpret_cast<TfLiteConvParams*>(tfLiteNode->builtin_data);
    if (!tfLiteNodeParameters)
    {
        // No Activation
        return kTfLiteOk;
    }
    // Check activation
    TfLiteFusedActivation activationType = tfLiteNodeParameters->activation;
    return FusedActivation(tfLiteContext, tfLiteNode, activationType, layer, 0, delegateData);

}

// Conv3d is only correctly supported for external delegates from TF Lite v2.6, as there was a breaking bug in v2.5.
#if defined(ARMNN_POST_TFLITE_2_5)
TfLiteStatus VisitConv3dOperator(DelegateData& delegateData,
                                 TfLiteContext* tfLiteContext,
                                 TfLiteNode* tfLiteNode,
                                 int nodeIndex,
                                 int32_t operatorCode)
{
    auto numInputs = tfLiteNode->inputs->size;
    if (numInputs < 2)
    {
        TF_LITE_MAYBE_KERNEL_LOG(
                tfLiteContext, "TfLiteArmnnDelegate: Minimum number of inputs (%d != %d) in node #%d",
                2, numInputs, nodeIndex);
        return kTfLiteError;
    }
    TF_LITE_ENSURE_STATUS(ValidateNumOutputs(tfLiteContext, tfLiteNode, 1, nodeIndex));

    armnn::Convolution3dDescriptor descriptor;
    const auto params = reinterpret_cast<TfLiteConv3DParams*>(tfLiteNode->builtin_data);

    bool biasEnabled = tfLiteNode->inputs->size == 3 ? true : false;
    descriptor.m_BiasEnabled = biasEnabled;
    descriptor.m_DataLayout = armnn::DataLayout::NDHWC;
    descriptor.m_StrideX = NonNegative(params->stride_width, nodeIndex);
    descriptor.m_StrideY = NonNegative(params->stride_height, nodeIndex);
    descriptor.m_StrideZ = NonNegative(params->stride_depth, nodeIndex);
    descriptor.m_DilationX = NonNegative(params->dilation_width_factor, nodeIndex);
    descriptor.m_DilationY = NonNegative(params->dilation_height_factor, nodeIndex);
    descriptor.m_DilationZ = NonNegative(params->dilation_depth_factor, nodeIndex);

    const TfLiteTensor* tfLiteTensors = tfLiteContext->tensors;
    const TfLiteTensor& tfLiteInputTensor = tfLiteTensors[tfLiteNode->inputs->data[0]];
    if (!IsValid(tfLiteContext, tfLiteInputTensor, operatorCode, nodeIndex))
    {
        return kTfLiteError;
    }

    const TfLiteTensor& tfLiteOutputTensor = tfLiteTensors[tfLiteNode->outputs->data[0]];
    if (!IsValid(tfLiteContext, tfLiteOutputTensor, operatorCode, nodeIndex))
    {
        return kTfLiteError;
    }

    const TfLiteTensor& tfLiteFilterTensor = tfLiteTensors[tfLiteNode->inputs->data[1]];
    if (!IsValid(tfLiteContext, tfLiteFilterTensor, operatorCode, nodeIndex))
    {
        return kTfLiteError;
    }

    const armnn::TensorInfo& inputTensorInfo  = GetTensorInfoForTfLiteTensor(tfLiteInputTensor);
    const armnn::TensorInfo& outputTensorInfo = GetTensorInfoForTfLiteTensor(tfLiteOutputTensor);

    armnn::TensorInfo filterTensorInfo = GetTensorInfoForTfLiteTensor(tfLiteFilterTensor);

    armnn::TensorInfo biasTensorInfo;
    if(biasEnabled)
    {
        const TfLiteTensor& tfLiteBiasTensor = tfLiteTensors[tfLiteNode->inputs->data[2]];
        if (!IsValid(tfLiteContext, tfLiteBiasTensor, operatorCode, nodeIndex))
        {
            return kTfLiteError;
        }
        biasTensorInfo = GetTensorInfoForTfLiteTensor(tfLiteBiasTensor);
    }
    else
    {
        biasTensorInfo = armnn::TensorInfo(armnn::TensorShape({1}), GetDataType(tfLiteInputTensor));
    }

    armnn::Optional<armnn::TensorInfo> optionalBiasInfo(biasTensorInfo);

    // TfLite uses NDHWC tensors
    const unsigned int inputDepth  = inputTensorInfo.GetShape()[1];
    const unsigned int inputHeight = inputTensorInfo.GetShape()[2];
    const unsigned int inputWidth  = inputTensorInfo.GetShape()[3];

    // Assuming the filter is DHWIO : Depth, Height, Width, OutputChannels, InputChannels
    const unsigned int filterDepth  = filterTensorInfo.GetShape()[0];
    const unsigned int filterHeight = filterTensorInfo.GetShape()[1];
    const unsigned int filterWidth  = filterTensorInfo.GetShape()[2];

    // Calculate padding
    CalcPadding(inputDepth, filterDepth, descriptor.m_StrideZ, descriptor.m_DilationZ,
                descriptor.m_PadFront, descriptor.m_PadBack, params->padding);
    CalcPadding(inputHeight, filterHeight, descriptor.m_StrideY, descriptor.m_DilationY,
                descriptor.m_PadTop, descriptor.m_PadBottom, params->padding);
    CalcPadding(inputWidth, filterWidth, descriptor.m_StrideX, descriptor.m_DilationX,
                descriptor.m_PadLeft, descriptor.m_PadRight, params->padding);

    // If the m_Network is a nullptr, this signals that a prerequisite TfLite callback is required to clarify the
    // support for the operator
    // If supported, VisitConvolutionOperator will be called again to add the layer to the network as seen below.
    if (!delegateData.m_Network)
    {
        bool isSupported = false;
        FORWARD_LAYER_SUPPORT_FUNC(__func__,
                                   tfLiteContext,
                                   IsConvolution3dSupported,
                                   delegateData.m_Backends,
                                   isSupported,
                                   inputTensorInfo,
                                   outputTensorInfo,
                                   descriptor,
                                   filterTensorInfo,
                                   optionalBiasInfo);
        return isSupported ? kTfLiteOk : kTfLiteError;
    }

    armnn::IConnectableLayer* layer =  delegateData.m_Network->AddConvolution3dLayer(descriptor);
    ARMNN_ASSERT(layer != nullptr);

    // Add a constant layer for weights and biases if inputs are constant,
    // which are connected to the Convolution3d layer as inputs.
    if (tflite::IsConstantTensor(&tfLiteFilterTensor))
    {
        auto filter = CreateConstTensor(&tfLiteFilterTensor,
                                        filterTensorInfo,
                                        armnn::Optional<armnn::PermutationVector&>());

        armnn::IConnectableLayer* weightsLayer = delegateData.m_Network->AddConstantLayer(filter);
        ARMNN_ASSERT(weightsLayer != nullptr);

        weightsLayer->GetOutputSlot(0).Connect(layer->GetInputSlot(1u));
        weightsLayer->GetOutputSlot(0).SetTensorInfo(filterTensorInfo);
    }

    if(biasEnabled)
    {
        const TfLiteTensor& tfLiteBiasTensor = tfLiteTensors[tfLiteNode->inputs->data[2]];
        if(tflite::IsConstantTensor(&tfLiteBiasTensor))
        {
            auto biases = CreateConstTensor(&tfLiteBiasTensor,
                                            biasTensorInfo,
                                            armnn::Optional<armnn::PermutationVector&>());

            armnn::IConnectableLayer* biasLayer = delegateData.m_Network->AddConstantLayer(biases);
            ARMNN_ASSERT(biasLayer != nullptr);

            biasLayer->GetOutputSlot(0).Connect(layer->GetInputSlot(2u));
            biasLayer->GetOutputSlot(0).SetTensorInfo(biasTensorInfo);
        }
    }

    armnn::IOutputSlot& outputSlot = layer->GetOutputSlot(0);
    outputSlot.SetTensorInfo(outputTensorInfo);

    Connect(layer, tfLiteNode, delegateData);

    auto* tfLiteNodeParameters = reinterpret_cast<TfLiteConv3DParams*>(tfLiteNode->builtin_data);
    if (!tfLiteNodeParameters)
    {
        // No Activation
        return kTfLiteOk;
    }

    // Check activation
    TfLiteFusedActivation activationType = tfLiteNodeParameters->activation;
    return FusedActivation(tfLiteContext, tfLiteNode, activationType, layer, 0, delegateData);
}
#endif

TfLiteStatus VisitDepthwiseConv2dOperator(DelegateData& delegateData,
                                          TfLiteContext* tfLiteContext,
                                          TfLiteNode* tfLiteNode,
                                          int nodeIndex,
                                          int32_t operatorCode)
{
    auto numInputs = tfLiteNode->inputs->size;
    if (numInputs < 2)
    {
        TF_LITE_MAYBE_KERNEL_LOG(
            tfLiteContext, "TfLiteArmnnDelegate: Minimum number of inputs (%d != %d) in node #%d",
            2, numInputs, nodeIndex);
        return kTfLiteError;
    }
    TF_LITE_ENSURE_STATUS(ValidateNumOutputs(tfLiteContext, tfLiteNode, 1, nodeIndex));

    bool biasEnabled = tfLiteNode->inputs->size > 2;

    armnn::DepthwiseConvolution2dDescriptor descriptor;
    const auto params = reinterpret_cast<TfLiteDepthwiseConvParams*>(tfLiteNode->builtin_data);

    descriptor.m_BiasEnabled = biasEnabled;
    descriptor.m_StrideX = NonNegative(params->stride_width, nodeIndex);
    descriptor.m_StrideY = NonNegative(params->stride_height, nodeIndex);
    descriptor.m_DataLayout = armnn::DataLayout::NHWC;
    descriptor.m_DilationX = NonNegative(params->dilation_width_factor, nodeIndex);
    descriptor.m_DilationY = NonNegative(params->dilation_height_factor, nodeIndex);

    const TfLiteTensor* tfLiteTensors = tfLiteContext->tensors;
    const TfLiteTensor& tfLiteInputTensor = tfLiteTensors[tfLiteNode->inputs->data[0]];
    if(!IsValid(&tfLiteInputTensor))
    {
        TF_LITE_MAYBE_KERNEL_LOG(
            tfLiteContext,
            "TfLiteArmnnDelegate: Invalid input tensor in operator #%d node #%d: ",
            operatorCode, nodeIndex);
        return kTfLiteError;
    }
    if (IsDynamicTensor(tfLiteInputTensor))
    {
        TF_LITE_MAYBE_KERNEL_LOG(
            tfLiteContext,
            "TfLiteArmnnDelegate: Dynamic input tensors are not supported in operator #%d node #%d: ",
            operatorCode, nodeIndex);
        return kTfLiteError;
    }
    const TfLiteTensor& tfLiteOutputTensor = tfLiteTensors[tfLiteNode->outputs->data[0]];
    if(!IsValid(&tfLiteOutputTensor))
    {
        TF_LITE_MAYBE_KERNEL_LOG(
            tfLiteContext,
            "TfLiteArmnnDelegate: Invalid output tensor in operator #%d node #%d: ",
            operatorCode, nodeIndex);
        return kTfLiteError;
    }
    if (IsDynamicTensor(tfLiteOutputTensor))
    {
        TF_LITE_MAYBE_KERNEL_LOG(
            tfLiteContext,
            "TfLiteArmnnDelegate: Dynamic output tensors are not supported in operator #%d node #%d: ",
            operatorCode, nodeIndex);
        return kTfLiteError;
    }

    const TfLiteTensor& tfLiteFilterTensor = tfLiteTensors[tfLiteNode->inputs->data[1]];
    if(!IsValid(&tfLiteFilterTensor))
    {
        TF_LITE_MAYBE_KERNEL_LOG(
            tfLiteContext,
            "TfLiteArmnnDelegate: Invalid filter tensor in operator #%d node #%d: ",
            operatorCode, nodeIndex);
        return kTfLiteError;
    }
    if (IsDynamicTensor(tfLiteFilterTensor))
    {
        TF_LITE_MAYBE_KERNEL_LOG(
            tfLiteContext,
            "TfLiteArmnnDelegate: Dynamic filter tensors are not supported in node #%d: ",
            nodeIndex);
        return kTfLiteError;
    }

    const armnn::TensorInfo& inputTensorInfo  = GetTensorInfoForTfLiteTensor(tfLiteInputTensor);
    const armnn::TensorInfo& outputTensorInfo = GetTensorInfoForTfLiteTensor(tfLiteOutputTensor);

    armnn::TensorInfo filterTensorInfo = GetTensorInfoForTfLiteTensor(tfLiteFilterTensor);

    // Assuming input is NHWC
    unsigned int inputHeight = inputTensorInfo.GetShape()[1];
    unsigned int inputWidth  = inputTensorInfo.GetShape()[2];

    // TensorflowLite weights come in the format [1, H, W, I * M]
    unsigned int filterHeight = filterTensorInfo.GetShape()[1];
    unsigned int filterWidth  = filterTensorInfo.GetShape()[2];

    // Calculate padding
    CalcPadding(inputHeight, filterHeight, descriptor.m_StrideY, descriptor.m_DilationY,
                descriptor.m_PadTop, descriptor.m_PadBottom, params->padding);
    CalcPadding(inputWidth, filterWidth, descriptor.m_StrideX, descriptor.m_DilationX,
                descriptor.m_PadLeft, descriptor.m_PadRight, params->padding);

    armnn::TensorInfo biasTensorInfo;
    if(biasEnabled)
    {
        const TfLiteTensor& tfLiteBiasTensor = tfLiteTensors[tfLiteNode->inputs->data[2]];
        if(!IsValid(&tfLiteBiasTensor))
        {
            TF_LITE_MAYBE_KERNEL_LOG(
                tfLiteContext,
                "TfLiteArmnnDelegate: Invalid bias tensor in operator #%d node #%d: ",
                operatorCode, nodeIndex);
            return kTfLiteError;
        }
        if (IsDynamicTensor(tfLiteBiasTensor))
        {
            TF_LITE_MAYBE_KERNEL_LOG(
                tfLiteContext,
                "TfLiteArmnnDelegate: Dynamic bias tensors are not supported in node #%d: ",
                nodeIndex);
            return kTfLiteError;
        }
        biasTensorInfo = GetTensorInfoForTfLiteTensor(tfLiteBiasTensor);
    }
    else
    {
        biasTensorInfo = armnn::TensorInfo(armnn::TensorShape({1}), GetDataType(tfLiteInputTensor));
    }

    // For depthwise the weights layout is the same as for tflite [1, H, W, I*M]. No permutation required.
    auto filter = CreateConstTensor(&tfLiteFilterTensor, filterTensorInfo);

    if (!delegateData.m_Network)
    {
        bool isSupported = false;
        FORWARD_LAYER_SUPPORT_FUNC(__func__,
                                   tfLiteContext,
                                   IsDepthwiseConvolutionSupported,
                                   delegateData.m_Backends,
                                   isSupported,
                                   inputTensorInfo,
                                   outputTensorInfo,
                                   descriptor,
                                   filter.GetInfo(),
                                   armnn::Optional<armnn::TensorInfo>(biasTensorInfo));
        return isSupported ? kTfLiteOk : kTfLiteError;
    }

    armnn::IConnectableLayer* layer = nullptr;

    if(biasEnabled)
    {
        auto biases =
            CreateConstTensor(&tfLiteContext->tensors[tfLiteNode->inputs->data[2]],
                              biasTensorInfo);
        layer = delegateData.m_Network->AddDepthwiseConvolution2dLayer(descriptor,
                                                                       filter,
                                                                       armnn::Optional<armnn::ConstTensor>(biases));
    }
    else
    {
        layer = delegateData.m_Network->AddDepthwiseConvolution2dLayer(descriptor,
                                                                       filter,
                                                                       armnn::EmptyOptional());
    }

    ARMNN_ASSERT(layer != nullptr);

    armnn::IOutputSlot& outputSlot = layer->GetOutputSlot(0);
    outputSlot.SetTensorInfo(outputTensorInfo);

    Connect(layer, tfLiteNode, delegateData);
    auto* tfLiteNodeParameters = reinterpret_cast<TfLiteDepthwiseConvParams*>(tfLiteNode->builtin_data);
    if (!tfLiteNodeParameters)
    {
        // No Activation
        return kTfLiteOk;
    }
    // Check activation
    TfLiteFusedActivation activationType = tfLiteNodeParameters->activation;
    return FusedActivation(tfLiteContext, tfLiteNode, activationType, layer, 0, delegateData);
}

TfLiteStatus VisitTransposeConv2dOperator(DelegateData& delegateData,
                                          TfLiteContext* tfLiteContext,
                                          TfLiteNode* tfLiteNode,
                                          int nodeIndex,
                                          int32_t operatorCode)
{
    TF_LITE_ENSURE_STATUS(ValidateNumInputs(tfLiteContext, tfLiteNode, 3, nodeIndex));
    TF_LITE_ENSURE_STATUS(ValidateNumOutputs(tfLiteContext, tfLiteNode, 1, nodeIndex));

    armnn::TransposeConvolution2dDescriptor descriptor;
    auto* parameters = reinterpret_cast<TfLiteTransposeConvParams*>(tfLiteNode->builtin_data);
    descriptor.m_BiasEnabled = false;
    descriptor.m_StrideX = NonNegative(parameters->stride_width, nodeIndex);
    descriptor.m_StrideY = NonNegative(parameters->stride_height, nodeIndex);
    descriptor.m_DataLayout = armnn::DataLayout::NHWC;

    const TfLiteTensor* tfLiteTensors = tfLiteContext->tensors;
    const TfLiteTensor& tfLiteOutputShapeTensor = tfLiteTensors[tfLiteNode->inputs->data[0]];
    if(!IsValid(&tfLiteOutputShapeTensor))
    {
        TF_LITE_MAYBE_KERNEL_LOG(
            tfLiteContext,
            "TfLiteArmnnDelegate: Invalid input tensor in operator #%d node #%d: ",
            operatorCode, nodeIndex);
        return kTfLiteError;
    }
    if (IsDynamicTensor(tfLiteOutputShapeTensor))
    {
        TF_LITE_MAYBE_KERNEL_LOG(
            tfLiteContext,
            "TfLiteArmnnDelegate: Dynamic input tensors are not supported in operator #%d node #%d: ",
            operatorCode, nodeIndex);
        return kTfLiteError;
    }

    armnn::TensorInfo tensorInfo = GetTensorInfoForTfLiteTensor(tfLiteOutputShapeTensor);
    std::vector<int32_t> outputShape(tensorInfo.GetNumElements());
    if (tensorInfo.GetDataType() == armnn::DataType::Signed32)
    {
        for(unsigned int i=0; i < tensorInfo.GetNumElements(); i++)
        {
            outputShape[i] = ::tflite::GetTensorData<int32_t>(&tfLiteOutputShapeTensor)[i];
        }
    }

    if (tensorInfo.GetDataType() == armnn::DataType::QAsymmU8)
    {
        for(unsigned int i=0; i < tensorInfo.GetNumElements(); i++)
        {
            outputShape[i] = ::tflite::GetTensorData<uint8_t>(&tfLiteOutputShapeTensor)[i];
        }
    }
    // Change from signed to unsigned int to store in TransposeConvolution2dDescriptor.
    for (int dimension : outputShape)
    {
        descriptor.m_OutputShape.push_back(static_cast<unsigned int>(dimension));
    }
    descriptor.m_OutputShapeEnabled = true;

    const TfLiteTensor& tfLiteInputTensor = tfLiteTensors[tfLiteNode->inputs->data[2]];
    if(!IsValid(&tfLiteInputTensor))
    {
        TF_LITE_MAYBE_KERNEL_LOG(
            tfLiteContext,
            "TfLiteArmnnDelegate: Invalid input tensor in operator #%d node #%d: ",
            operatorCode, nodeIndex);
        return kTfLiteError;
    }
    if (IsDynamicTensor(tfLiteInputTensor))
    {
        TF_LITE_MAYBE_KERNEL_LOG(
            tfLiteContext,
            "TfLiteArmnnDelegate: Dynamic input tensors are not supported in operator #%d node #%d: ",
            operatorCode, nodeIndex);
        return kTfLiteError;
    }

    const TfLiteTensor& tfLiteOutputTensor = tfLiteTensors[tfLiteNode->outputs->data[0]];
    if(!IsValid(&tfLiteOutputTensor))
    {
        TF_LITE_MAYBE_KERNEL_LOG(
            tfLiteContext,
            "TfLiteArmnnDelegate: Invalid output tensor in operator #%d node #%d: ",
            operatorCode, nodeIndex);
        return kTfLiteError;
    }
    if (IsDynamicTensor(tfLiteOutputTensor))
    {
        TF_LITE_MAYBE_KERNEL_LOG(
            tfLiteContext,
            "TfLiteArmnnDelegate: Dynamic output tensors are not supported in operator #%d node #%d: ",
            operatorCode, nodeIndex);
        return kTfLiteError;
    }

    const TfLiteTensor& tfLiteFilterTensor = tfLiteTensors[tfLiteNode->inputs->data[1]];
    if(!IsValid(&tfLiteFilterTensor))
    {
        TF_LITE_MAYBE_KERNEL_LOG(
            tfLiteContext,
            "TfLiteArmnnDelegate: Invalid filter tensor in operator #%d node #%d: ",
            operatorCode, nodeIndex);
        return kTfLiteError;
    }
    if (IsDynamicTensor(tfLiteFilterTensor))
    {
        TF_LITE_MAYBE_KERNEL_LOG(
            tfLiteContext,
            "TfLiteArmnnDelegate: Dynamic filter tensors are not supported in operator #%d node #%d: ",
            operatorCode, nodeIndex);
        return kTfLiteError;
    }

    const armnn::TensorInfo& inputTensorInfo  = GetTensorInfoForTfLiteTensor(tfLiteInputTensor);
    const armnn::TensorInfo& outputTensorInfo = GetTensorInfoForTfLiteTensor(tfLiteOutputTensor);
    armnn::TensorInfo filterTensorInfo = GetTensorInfoForTfLiteTensor(tfLiteFilterTensor);

    // TfLite uses NHWC tensors
    const unsigned int inputHeight = inputTensorInfo.GetShape()[1];
    const unsigned int inputWidth  = inputTensorInfo.GetShape()[2];

    const unsigned int filterHeight = filterTensorInfo.GetShape()[1];
    const unsigned int filterWidth  = filterTensorInfo.GetShape()[2];

    // Calculate padding
    CalcPadding(inputHeight,
                filterHeight,
                descriptor.m_StrideY,
                1, // dilation y
                descriptor.m_PadTop,
                descriptor.m_PadBottom,
                parameters->padding);
    CalcPadding(inputWidth,
                filterWidth,
                descriptor.m_StrideX,
                1, // dilation x
                descriptor.m_PadLeft,
                descriptor.m_PadRight,
                parameters->padding);

    // Set up filter
    auto filterTensor = CreateConstTensor(&tfLiteFilterTensor,
                                          filterTensorInfo,
                                          armnn::Optional<armnn::PermutationVector&>());
    if (!delegateData.m_Network)
    {
        bool isSupported = false;
        FORWARD_LAYER_SUPPORT_FUNC(__func__,
                                   tfLiteContext,
                                   IsTransposeConvolution2dSupported,
                                   delegateData.m_Backends,
                                   isSupported,
                                   inputTensorInfo,
                                   outputTensorInfo,
                                   descriptor,
                                   filterTensorInfo,
                                   armnn::EmptyOptional());
        return isSupported ? kTfLiteOk : kTfLiteError;
    }

    armnn::IConnectableLayer* layer = delegateData.m_Network->AddTransposeConvolution2dLayer(descriptor,
                                                                                             filterTensor,
                                                                                             armnn::EmptyOptional());
    ARMNN_ASSERT(layer != nullptr);

    armnn::IOutputSlot& outputSlot = layer->GetOutputSlot(0);
    outputSlot.SetTensorInfo(outputTensorInfo);

    // Connect
    if (delegateData.m_OutputSlotForNode[static_cast<unsigned int>(tfLiteNode->inputs->data[2])] != nullptr)
    {
        delegateData.m_OutputSlotForNode[static_cast<unsigned int>(tfLiteNode->inputs->data[2])]->
                                                                   Connect(layer->GetInputSlot(0));
    }

    // Prepare output slots
    for (unsigned int outputIndex = 0; outputIndex < layer->GetNumOutputSlots(); ++outputIndex)
    {
        armnn::IOutputSlot& outputSlot = layer->GetOutputSlot(outputIndex);
        delegateData.m_OutputSlotForNode[static_cast<unsigned int>(tfLiteNode->outputs->data[outputIndex])] =
                                                                   &outputSlot;
    }
    return kTfLiteOk;
}

TfLiteStatus VisitConvolutionOperator(DelegateData& delegateData,
                                      TfLiteContext* tfLiteContext,
                                      TfLiteNode* tfLiteNode,
                                      int nodeIndex,
                                      int32_t operatorCode)
{
    switch(operatorCode)
    {
        case kTfLiteBuiltinConv2d:
            return VisitConv2dOperator(delegateData, tfLiteContext, tfLiteNode, nodeIndex, operatorCode);
// Conv3d is only correctly supported for external delegates from TF Lite v2.6, as there was a breaking bug in v2.5.
#if defined(ARMNN_POST_TFLITE_2_5)
        case kTfLiteBuiltinConv3d:
            return VisitConv3dOperator(delegateData, tfLiteContext, tfLiteNode, nodeIndex, operatorCode);
#endif
        case kTfLiteBuiltinDepthwiseConv2d:
            return VisitDepthwiseConv2dOperator(delegateData, tfLiteContext, tfLiteNode, nodeIndex, operatorCode);
        case kTfLiteBuiltinTransposeConv:
            return VisitTransposeConv2dOperator(delegateData, tfLiteContext, tfLiteNode, nodeIndex, operatorCode);
        default:
            return kTfLiteError;
    }
}

} // namespace armnnDelegate
