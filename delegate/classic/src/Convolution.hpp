//
// Copyright © 2022-2024 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include <ClassicDelegateUtils.hpp>
#include <SharedFunctions.hpp>

#include <tensorflow/lite/builtin_ops.h>
#include <tensorflow/lite/c/builtin_op_data.h>
#include <tensorflow/lite/c/common.h>
#include <tensorflow/lite/minimal_logging.h>
#include <tensorflow/lite/kernels/internal/tensor.h>
#include <armnnUtils/DataLayoutIndexed.hpp>

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

    bool biasEnabled = IsOptionalOperandPresent(tfLiteNode, 2);
    descriptor.m_BiasEnabled = biasEnabled;
    descriptor.m_StrideX = NonNegative(params->stride_width, nodeIndex);
    descriptor.m_StrideY = NonNegative(params->stride_height, nodeIndex);
    descriptor.m_DataLayout = armnn::DataLayout::NHWC;
    descriptor.m_DilationX = NonNegative(params->dilation_width_factor, nodeIndex);
    descriptor.m_DilationY = NonNegative(params->dilation_height_factor, nodeIndex);

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
    const armnn::TensorInfo& outputTensorInfo = GetTensorInfoForTfLiteTensor(tfLiteOutputTensor, true);

    auto* tfLiteNodeParameters = reinterpret_cast<TfLiteConvParams*>(tfLiteNode->builtin_data);
    TfLiteFusedActivation activationType=kTfLiteActNone;
    if (tfLiteNodeParameters)
    {
        activationType = tfLiteNodeParameters->activation;
        TfLiteStatus activationStatus = ValidateFusedActivationOperator(delegateData, tfLiteContext, outputTensorInfo,
                                                                        outputTensorInfo, activationType);
        if(activationStatus != kTfLiteOk)
        {
            return kTfLiteError;
        }

    }

    const armnn::TensorInfo& filterTensorInfo = GetTensorInfoForTfLiteTensor(tfLiteFilterTensor);

    // Check for unsupported group convolution
    if (IsGroupedConvolution(inputTensorInfo.GetShape(), filterTensorInfo.GetShape(), descriptor.m_DataLayout))
    {
        TF_LITE_MAYBE_KERNEL_LOG(
                tfLiteContext,
                "TfLiteArmnnDelegate: Group convolution is not supported.");
        return kTfLiteError;
    }

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

    armnn::BackendId setBackend;
    if (!delegateData.m_Network)
    {
        bool filterIsConst = filterTensorInfo.IsConstant();

        if (!filterIsConst)
        {
            filterIsConst = WillInputBeOptimizedToConst(tfLiteContext, tfLiteNode->inputs->data[1]);
        }
        armnn::TensorInfo filterTensorInfoCopy(filterTensorInfo);
        filterTensorInfoCopy.SetConstant(filterIsConst);
        armnn::Optional<armnn::TensorInfo> optionalBiasInfoCopy(biasTensorInfo);

        if (biasEnabled)
        {
            bool biasIsConst = biasTensorInfo.IsConstant();

            if (!biasIsConst)
            {
                biasIsConst = WillInputBeOptimizedToConst(tfLiteContext, tfLiteNode->inputs->data[2]);
            }
            optionalBiasInfoCopy.value().SetConstant(biasIsConst);
        }

        bool isSupported = false;
        FORWARD_LAYER_SUPPORT_FUNC("CONV2D",
                                   tfLiteContext,
                                   IsConvolution2dSupported,
                                   delegateData.m_Backends,
                                   isSupported,
                                   setBackend,
                                   inputTensorInfo,
                                   outputTensorInfo,
                                   descriptor,
                                   filterTensorInfoCopy,
                                   optionalBiasInfoCopy);
        return isSupported ? kTfLiteOk : kTfLiteError;
    }

    // Set up filter and biases
    auto layerName = GetLayerName(armnn::LayerType::Convolution2d, nodeIndex);
    armnn::IConnectableLayer* layer = delegateData.m_Network->AddConvolution2dLayer(descriptor, layerName.c_str());
    layer->SetBackendId(setBackend);

    if (filterTensorInfo.IsConstant())
    {
        auto filter = CreateConstTensor(&tfLiteContext->tensors[tfLiteNode->inputs->data[1]], filterTensorInfo);

        auto filterName = GetLayerName(armnn::LayerType::Constant, nodeIndex, "Filter");
        armnn::IConnectableLayer* weightsLayer = delegateData.m_Network->AddConstantLayer(filter, filterName.c_str());
        weightsLayer->GetOutputSlot(0).Connect(layer->GetInputSlot(1u));
        weightsLayer->GetOutputSlot(0).SetTensorInfo(filterTensorInfo);
    }

    if (biasEnabled)
    {
        const TfLiteTensor& tfLiteBiasTensor = tfLiteTensors[tfLiteNode->inputs->data[2]];
        if(biasTensorInfo.IsConstant())
        {
            auto biasTensor = CreateConstTensor(&tfLiteBiasTensor, biasTensorInfo);

            auto biasName = GetLayerName(armnn::LayerType::Constant, nodeIndex, "Bias");
            armnn::IConnectableLayer* biasLayer = delegateData.m_Network->AddConstantLayer(biasTensor,
                                                                                           biasName.c_str());
            ARMNN_ASSERT(biasLayer != nullptr);
            biasLayer->GetOutputSlot(0).Connect(layer->GetInputSlot(2u));
            biasLayer->GetOutputSlot(0).SetTensorInfo(biasTensorInfo);
        }
    }

    // The data input can also be constant, so we must check that this is also allocated to an input slot
    if (inputTensorInfo.IsConstant())
    {
        auto input = CreateConstTensor(&tfLiteContext->tensors[tfLiteNode->inputs->data[0]], inputTensorInfo);

        auto inputName = GetLayerName(armnn::LayerType::Constant, nodeIndex, "Input");
        armnn::IConnectableLayer* inputLayer = delegateData.m_Network->AddConstantLayer(input, inputName.c_str());
        inputLayer->GetOutputSlot(0).Connect(layer->GetInputSlot(0u));
        inputLayer->GetOutputSlot(0).SetTensorInfo(inputTensorInfo);
    }

    ARMNN_ASSERT(layer != nullptr);

    armnn::IOutputSlot& outputSlot = layer->GetOutputSlot(0);
    outputSlot.SetTensorInfo(outputTensorInfo);

    if(Connect(layer, tfLiteNode, delegateData) != kTfLiteOk)
    {
        return kTfLiteError;
    }

    if (!tfLiteNodeParameters)
    {
        // No Activation
        return kTfLiteOk;
    }

    // Check and Create activation
    return FusedActivation(tfLiteContext, tfLiteNode, activationType, layer, 0, delegateData, nodeIndex);
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

    bool biasEnabled = IsOptionalOperandPresent(tfLiteNode, 2);
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
    const armnn::TensorInfo& outputTensorInfo = GetTensorInfoForTfLiteTensor(tfLiteOutputTensor, true);

    auto* tfLiteNodeParameters = reinterpret_cast<TfLiteConv3DParams*>(tfLiteNode->builtin_data);
    TfLiteFusedActivation activationType=kTfLiteActNone;
    if (tfLiteNodeParameters)
    {
        activationType = tfLiteNodeParameters->activation;
        TfLiteStatus activationStatus = ValidateFusedActivationOperator(delegateData, tfLiteContext, outputTensorInfo,
                                                                        outputTensorInfo, activationType);
        if(activationStatus != kTfLiteOk)
        {
            return kTfLiteError;
        }

    }

    const armnn::TensorInfo& filterTensorInfo = GetTensorInfoForTfLiteTensor(tfLiteFilterTensor);

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
    armnn::BackendId setBackend;
    if (!delegateData.m_Network)
    {
        bool isSupported = false;
        FORWARD_LAYER_SUPPORT_FUNC("CONV3D",
                                   tfLiteContext,
                                   IsConvolution3dSupported,
                                   delegateData.m_Backends,
                                   isSupported,
                                   setBackend,
                                   inputTensorInfo,
                                   outputTensorInfo,
                                   descriptor,
                                   filterTensorInfo,
                                   optionalBiasInfo);
        return isSupported ? kTfLiteOk : kTfLiteError;
    }

    auto layerName = GetLayerName(armnn::LayerType::Convolution3d, nodeIndex);
    armnn::IConnectableLayer* layer =  delegateData.m_Network->AddConvolution3dLayer(descriptor, layerName.c_str());
    layer->SetBackendId(setBackend);
    ARMNN_ASSERT(layer != nullptr);

    // Add a constant layer for weights and biases if inputs are constant,
    // which are connected to the Convolution3d layer as inputs.
    if (filterTensorInfo.IsConstant())
    {
        auto filter = CreateConstTensor(&tfLiteFilterTensor, filterTensorInfo);

        auto filterName = GetLayerName(armnn::LayerType::Constant, nodeIndex, "Filter");
        armnn::IConnectableLayer* weightsLayer = delegateData.m_Network->AddConstantLayer(filter, filterName.c_str());
        ARMNN_ASSERT(weightsLayer != nullptr);

        weightsLayer->GetOutputSlot(0).Connect(layer->GetInputSlot(1u));
        weightsLayer->GetOutputSlot(0).SetTensorInfo(filterTensorInfo);
    }

    if(biasEnabled)
    {
        const TfLiteTensor& tfLiteBiasTensor = tfLiteTensors[tfLiteNode->inputs->data[2]];
        if(biasTensorInfo.IsConstant())
        {
            auto biases = CreateConstTensor(&tfLiteBiasTensor, biasTensorInfo);

            auto biasName = GetLayerName(armnn::LayerType::Constant, nodeIndex, "Bias");
            armnn::IConnectableLayer* biasLayer = delegateData.m_Network->AddConstantLayer(biases, biasName.c_str());
            ARMNN_ASSERT(biasLayer != nullptr);

            biasLayer->GetOutputSlot(0).Connect(layer->GetInputSlot(2u));
            biasLayer->GetOutputSlot(0).SetTensorInfo(biasTensorInfo);
        }
    }

    // The data input can also be constant, so we must check that this is also allocated to an input slot
    if(inputTensorInfo.IsConstant())
    {
        auto input = CreateConstTensor(&tfLiteContext->tensors[tfLiteNode->inputs->data[0]], inputTensorInfo);

        auto inputName = GetLayerName(armnn::LayerType::Constant, nodeIndex, "Input");
        armnn::IConnectableLayer* inputLayer = delegateData.m_Network->AddConstantLayer(input, inputName.c_str());
        inputLayer->GetOutputSlot(0).Connect(layer->GetInputSlot(0u));
        inputLayer->GetOutputSlot(0).SetTensorInfo(inputTensorInfo);
    }

    armnn::IOutputSlot& outputSlot = layer->GetOutputSlot(0);
    outputSlot.SetTensorInfo(outputTensorInfo);

    if(Connect(layer, tfLiteNode, delegateData) != kTfLiteOk)
    {
        return kTfLiteError;
    }

    if (!tfLiteNodeParameters)
    {
        // No Activation
        return kTfLiteOk;
    }

    // Check and create activation
    return FusedActivation(tfLiteContext, tfLiteNode, activationType, layer, 0, delegateData, nodeIndex);
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

    bool biasEnabled = IsOptionalOperandPresent(tfLiteNode, 2);

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
    const armnn::TensorInfo& outputTensorInfo = GetTensorInfoForTfLiteTensor(tfLiteOutputTensor, true);

    auto* tfLiteNodeParameters = reinterpret_cast<TfLiteDepthwiseConvParams *>(tfLiteNode->builtin_data);
    TfLiteFusedActivation activationType = kTfLiteActNone;
    if (tfLiteNodeParameters)
    {
        activationType = tfLiteNodeParameters->activation;
        TfLiteStatus activationStatus = ValidateFusedActivationOperator(delegateData, tfLiteContext, outputTensorInfo,
                                                                        outputTensorInfo, activationType);
        if(activationStatus != kTfLiteOk)
        {
            return kTfLiteError;
        }

    }

    const armnn::TensorInfo& filterTensorInfo = GetTensorInfoForTfLiteTensor(tfLiteFilterTensor);

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

    armnn::BackendId setBackend;
    if (!delegateData.m_Network)
    {
        bool filterIsConst = filterTensorInfo.IsConstant();

        if (!filterIsConst)
        {
            filterIsConst = WillInputBeOptimizedToConst(tfLiteContext, tfLiteNode->inputs->data[1]);
        }
        armnn::TensorInfo filterTensorInfoCopy(filterTensorInfo);
        filterTensorInfoCopy.SetConstant(filterIsConst);

        armnn::Optional<armnn::TensorInfo> optionalBiasInfoCopy(biasTensorInfo);

        if (biasEnabled)
        {
            bool biasIsConst = biasTensorInfo.IsConstant();

            if (!biasIsConst)
            {
                biasIsConst = WillInputBeOptimizedToConst(tfLiteContext, tfLiteNode->inputs->data[2]);
            }
            optionalBiasInfoCopy.value().SetConstant(biasIsConst);
        }

        bool isSupported = false;
        FORWARD_LAYER_SUPPORT_FUNC("DEPTHWISE_CONV2D",
                                   tfLiteContext,
                                   IsDepthwiseConvolutionSupported,
                                   delegateData.m_Backends,
                                   isSupported,
                                   setBackend,
                                   inputTensorInfo,
                                   outputTensorInfo,
                                   descriptor,
                                   filterTensorInfoCopy,
                                   optionalBiasInfoCopy);
        return isSupported ? kTfLiteOk : kTfLiteError;
    }

    auto layerName = GetLayerName(armnn::LayerType::DepthwiseConvolution2d, nodeIndex);
    armnn::IConnectableLayer* layer = delegateData.m_Network->AddDepthwiseConvolution2dLayer(descriptor,
                                                                                             layerName.c_str());
    layer->SetBackendId(setBackend);

    if(filterTensorInfo.IsConstant())
    {
        // For depthwise the weights layout is the same as for tflite [1, H, W, I*M]. No permutation required.
        auto filter = CreateConstTensor(&tfLiteFilterTensor, filterTensorInfo);

        auto filterName = GetLayerName(armnn::LayerType::Constant, nodeIndex, "Filter");
        armnn::IConnectableLayer* weightsLayer = delegateData.m_Network->AddConstantLayer(filter, filterName.c_str());
        weightsLayer->GetOutputSlot(0).Connect(layer->GetInputSlot(1u));
        weightsLayer->GetOutputSlot(0).SetTensorInfo(filterTensorInfo);
    }

    if (biasEnabled)
    {
        const TfLiteTensor& tfLiteBiasTensor = tfLiteTensors[tfLiteNode->inputs->data[2]];
        if(biasTensorInfo.IsConstant())
        {
            auto biasTensor = CreateConstTensor(&tfLiteBiasTensor, biasTensorInfo);

            auto biasName = GetLayerName(armnn::LayerType::Constant, nodeIndex, "Bias");
            armnn::IConnectableLayer* biasLayer = delegateData.m_Network->AddConstantLayer(biasTensor,
                                                                                           biasName.c_str());
            ARMNN_ASSERT(biasLayer != nullptr);
            biasLayer->GetOutputSlot(0).Connect(layer->GetInputSlot(2u));
            biasLayer->GetOutputSlot(0).SetTensorInfo(biasTensorInfo);
        }
    }

    // The data input can also be constant, so we must check that this is also allocated to an input slot
    if(inputTensorInfo.IsConstant())
    {
        auto input = CreateConstTensor(&tfLiteContext->tensors[tfLiteNode->inputs->data[0]], inputTensorInfo);

        auto inputName = GetLayerName(armnn::LayerType::Constant, nodeIndex, "Input");
        armnn::IConnectableLayer* inputLayer = delegateData.m_Network->AddConstantLayer(input, inputName.c_str());
        inputLayer->GetOutputSlot(0).Connect(layer->GetInputSlot(0u));
        inputLayer->GetOutputSlot(0).SetTensorInfo(inputTensorInfo);
    }

    ARMNN_ASSERT(layer != nullptr);

    armnn::IOutputSlot& outputSlot = layer->GetOutputSlot(0);
    outputSlot.SetTensorInfo(outputTensorInfo);

    if(Connect(layer, tfLiteNode, delegateData) != kTfLiteOk)
    {
        return kTfLiteError;
    }

    if (!tfLiteNodeParameters)
    {
        // No Activation
        return kTfLiteOk;
    }
    // Check and create activation
    return FusedActivation(tfLiteContext, tfLiteNode, activationType, layer, 0, delegateData, nodeIndex);
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
    if (!IsValid(tfLiteContext, tfLiteOutputShapeTensor, operatorCode, nodeIndex))
    {
        return kTfLiteError;
    }

    const TfLiteTensor& tfLiteInputTensor = tfLiteTensors[tfLiteNode->inputs->data[2]];
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
    const armnn::TensorInfo& outputTensorInfo = GetTensorInfoForTfLiteTensor(tfLiteOutputTensor, true);
    const armnn::TensorInfo& filterTensorInfo = GetTensorInfoForTfLiteTensor(tfLiteFilterTensor);

    TfLiteFusedActivation activationType = kTfLiteActNone;
    if (parameters->activation != kTfLiteActNone)
    {
        activationType = parameters->activation;

        TfLiteStatus activationStatus = ValidateFusedActivationOperator(delegateData,
                                                                        tfLiteContext,
                                                                        outputTensorInfo,
                                                                        outputTensorInfo,
                                                                        activationType);
        if(activationStatus != kTfLiteOk)
        {
            return kTfLiteError;
        }
    }

    // TfLite uses NHWC tensors
    const unsigned int inputHeight = inputTensorInfo.GetShape()[1];
    const unsigned int inputWidth  = inputTensorInfo.GetShape()[2];

    const unsigned int filterHeight = filterTensorInfo.GetShape()[1];
    const unsigned int filterWidth  = filterTensorInfo.GetShape()[2];

    // This block determines the output shape of the transpose convolution.
    // If the output shape tensor is a constant, we can access the data at load time and set the shape of the layer.
    // If this is not constant, we do not have access to the shape data, so we have to use infer output shape.
    if (tflite::IsConstantTensor(&tfLiteOutputShapeTensor))
    {
        const armnn::TensorInfo outputShapeTensorInfo = GetTensorInfoForTfLiteTensor(tfLiteOutputShapeTensor);
        std::vector<int32_t> outputShape(outputShapeTensorInfo.GetNumElements());
        if (outputShapeTensorInfo.GetDataType() == armnn::DataType::Signed32)
        {
            for(unsigned int i=0; i < outputShapeTensorInfo.GetNumElements(); ++i)
            {
                outputShape[i] = ::tflite::GetTensorData<int32_t>(&tfLiteOutputShapeTensor)[i];
            }
        }

        if (outputShapeTensorInfo.GetDataType() == armnn::DataType::QAsymmU8)
        {
            for(unsigned int i=0; i < outputShapeTensorInfo.GetNumElements(); ++i)
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

        // TfLite uses NHWC tensors
        const unsigned int outputHeight = descriptor.m_OutputShape[1];
        const unsigned int outputWidth  = descriptor.m_OutputShape[2];

        CalcPadding(inputHeight,
                    filterHeight,
                    descriptor.m_StrideY,
                    1, // DilationY
                    descriptor.m_PadTop,
                    descriptor.m_PadBottom,
                    parameters->padding,
                    outputHeight);

        CalcPadding(inputWidth,
                    filterWidth,
                    descriptor.m_StrideX,
                    1, // DilationX
                    descriptor.m_PadLeft,
                    descriptor.m_PadRight,
                    parameters->padding,
                    outputWidth);
    }
    else
    {
        CalcPadding(inputHeight,
                    filterHeight,
                    descriptor.m_StrideY,
                    1, // DilationY
                    descriptor.m_PadTop,
                    descriptor.m_PadBottom,
                    parameters->padding);

        CalcPadding(inputWidth,
                    filterWidth,
                    descriptor.m_StrideX,
                    1, // DilationX
                    descriptor.m_PadLeft,
                    descriptor.m_PadRight,
                    parameters->padding);
    }

    // Set up filter
    auto filterTensor = CreateConstTensor(&tfLiteFilterTensor,
                                          filterTensorInfo);
    armnn::BackendId setBackend;
    if (!delegateData.m_Network)
    {
        bool isSupported = false;
        FORWARD_LAYER_SUPPORT_FUNC("TRANSPOSE_CONV2D",
                                   tfLiteContext,
                                   IsTransposeConvolution2dSupported,
                                   delegateData.m_Backends,
                                   isSupported,
                                   setBackend,
                                   inputTensorInfo,
                                   outputTensorInfo,
                                   descriptor,
                                   filterTensorInfo,
                                   armnn::EmptyOptional());
        return isSupported ? kTfLiteOk : kTfLiteError;
    }

    auto layerName = GetLayerName(armnn::LayerType::TransposeConvolution2d, nodeIndex);
    armnn::IConnectableLayer* layer = delegateData.m_Network->AddTransposeConvolution2dLayer(descriptor,
                                                                                             filterTensor,
                                                                                             armnn::EmptyOptional(),
                                                                                             layerName.c_str());
    layer->SetBackendId(setBackend);
    ARMNN_ASSERT(layer != nullptr);

    // The data input can be constant, so we must check that this is allocated to an input slot
    if(inputTensorInfo.IsConstant())
    {
        auto input = CreateConstTensor(&tfLiteContext->tensors[tfLiteNode->inputs->data[2]], inputTensorInfo);

        auto inputName = GetLayerName(armnn::LayerType::Constant, nodeIndex, "Input");
        armnn::IConnectableLayer* inputLayer = delegateData.m_Network->AddConstantLayer(input, inputName.c_str());
        inputLayer->GetOutputSlot(0).Connect(layer->GetInputSlot(0u));
        inputLayer->GetOutputSlot(0).SetTensorInfo(inputTensorInfo);
    }

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

    if (activationType != kTfLiteActNone)
    {
        return FusedActivation(tfLiteContext, tfLiteNode, activationType, layer, 0, delegateData, nodeIndex);
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
