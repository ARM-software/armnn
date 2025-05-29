//
// Copyright © 2023-2025 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include <OpaqueDelegateUtils.hpp>
#include <SharedFunctions.hpp>

#include <tensorflow/lite/builtin_ops.h>
#include <tensorflow/lite/c/builtin_op_data.h>
#include <tensorflow/lite/c/common.h>
#include <tensorflow/lite/minimal_logging.h>
#include <armnnUtils/DataLayoutIndexed.hpp>

namespace armnnOpaqueDelegate
{

TfLiteStatus VisitConv2dOperator(DelegateData& delegateData,
                                 TfLiteOpaqueContext* tfLiteContext,
                                 TfLiteOpaqueNode* tfLiteNode,
                                 int nodeIndex,
                                 int32_t operatorCode)
{
    auto numInputs = TfLiteOpaqueNodeNumberOfInputs(tfLiteNode);
    if (numInputs < 2)
    {
        TF_LITE_OPAQUE_MAYBE_KERNEL_LOG(
                tfLiteContext,
                "TfLiteArmnnOpaqueDelegate: Minimum number of inputs (%d != %d) in node #%d",
                2, numInputs, nodeIndex);
        return kTfLiteError;
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

    // Use input indices to get filter tensor.
    const TfLiteOpaqueTensor* tfLiteFilterTensor = TfLiteOpaqueContextGetOpaqueTensor(tfLiteContext, inputTensors[1]);
    if (!IsValid(tfLiteContext, tfLiteFilterTensor, operatorCode, nodeIndex))
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

    const armnn::TensorInfo& inputTensorInfo  = GetTensorInfoForTfLiteOpaqueTensor(tfLiteInputTensor);
    const armnn::TensorInfo& filterTensorInfo = GetTensorInfoForTfLiteOpaqueTensor(tfLiteFilterTensor);
    const armnn::TensorInfo& outputTensorInfo = GetTensorInfoForTfLiteOpaqueTensor(tfLiteOutputTensor, true);

    auto* tfLiteNodeParameters = reinterpret_cast<TfLiteConvParams*>(TfLiteOpaqueNodeGetBuiltinData(tfLiteNode));
    TfLiteFusedActivation activationType = kTfLiteActNone;
    if (tfLiteNodeParameters)
    {
        activationType = tfLiteNodeParameters->activation;
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

    armnn::TensorInfo biasTensorInfo;
    const TfLiteOpaqueTensor* tfLiteBiasTensor = nullptr;

    bool biasEnabled = IsOptionalOperandPresent(tfLiteNode, 2);
    if(biasEnabled)
    {
        // Use input indices to get bias tensor.
        tfLiteBiasTensor = TfLiteOpaqueContextGetOpaqueTensor(tfLiteContext, inputTensors[2]);
        if (!IsValid(tfLiteContext, tfLiteBiasTensor, operatorCode, nodeIndex))
        {
            return kTfLiteError;
        }
        biasTensorInfo = GetTensorInfoForTfLiteOpaqueTensor(tfLiteBiasTensor);
    }
    else
    {
        biasTensorInfo = armnn::TensorInfo(armnn::TensorShape({1}), GetDataType(tfLiteInputTensor));
    }

    armnn::Optional<armnn::TensorInfo> optionalBiasInfo(biasTensorInfo);

    armnn::Convolution2dDescriptor descriptor;
    descriptor.m_BiasEnabled = biasEnabled;
    descriptor.m_StrideX = NonNegative(tfLiteNodeParameters->stride_width, nodeIndex);
    descriptor.m_StrideY = NonNegative(tfLiteNodeParameters->stride_height, nodeIndex);
    descriptor.m_DataLayout = armnn::DataLayout::NHWC;
    descriptor.m_DilationX = NonNegative(tfLiteNodeParameters->dilation_width_factor, nodeIndex);
    descriptor.m_DilationY = NonNegative(tfLiteNodeParameters->dilation_height_factor, nodeIndex);

    // TfLite uses NHWC tensors
    const unsigned int inputHeight = inputTensorInfo.GetShape()[1];
    const unsigned int inputWidth  = inputTensorInfo.GetShape()[2];

    const unsigned int filterHeight = filterTensorInfo.GetShape()[1];
    const unsigned int filterWidth  = filterTensorInfo.GetShape()[2];

    // Check for unsupported group convolution
    if (IsGroupedConvolution(inputTensorInfo.GetShape(), filterTensorInfo.GetShape(), descriptor.m_DataLayout))
    {
        TF_LITE_OPAQUE_MAYBE_KERNEL_LOG(
                tfLiteContext,
                "TfLiteArmnnOpaqueDelegate: Group convolution is not supported.");
        return kTfLiteError;
    }

    // Calculate padding
    CalcPadding(inputHeight, filterHeight, descriptor.m_StrideY, descriptor.m_DilationY,
                descriptor.m_PadTop, descriptor.m_PadBottom, tfLiteNodeParameters->padding);
    CalcPadding(inputWidth, filterWidth, descriptor.m_StrideX, descriptor.m_DilationX,
                descriptor.m_PadLeft, descriptor.m_PadRight, tfLiteNodeParameters->padding);

    armnn::BackendId setBackend;
    if (!delegateData.m_Network)
    {
        bool filterIsConst = filterTensorInfo.IsConstant();

        if (!filterIsConst)
        {
            filterIsConst = WillInputBeOptimizedToConst(tfLiteContext, inputTensors[1]);
        }
        armnn::TensorInfo filterTensorInfoCopy(filterTensorInfo);
        filterTensorInfoCopy.SetConstant(filterIsConst);
        armnn::Optional<armnn::TensorInfo> optionalBiasInfoCopy(biasTensorInfo);

        if (biasEnabled)
        {
            bool biasIsConst = biasTensorInfo.IsConstant();

            if (!biasIsConst)
            {
                biasIsConst = WillInputBeOptimizedToConst(tfLiteContext, inputTensors[2]);
            }
            optionalBiasInfoCopy.value().SetConstant(biasIsConst);
        }

        bool isSupported = false;
        FORWARD_LAYER_OPAQUE_SUPPORT_FUNC("CONV2D",
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
    auto layerName = GetName(armnn::LayerType::Convolution2d, nodeIndex);
    armnn::IConnectableLayer* layer = delegateData.m_Network->AddConvolution2dLayer(descriptor, layerName.c_str());
    layer->SetBackendId(setBackend);

    if(filterTensorInfo.IsConstant())
    {
        auto filter = CreateConstTensor(tfLiteFilterTensor, filterTensorInfo);

        auto filterName = GetName(armnn::LayerType::Constant, nodeIndex, "Filter");
        armnn::IConnectableLayer* weightsLayer = delegateData.m_Network->AddConstantLayer(filter, filterName.c_str());
        weightsLayer->GetOutputSlot(0).Connect(layer->GetInputSlot(1u));
        weightsLayer->GetOutputSlot(0).SetTensorInfo(filterTensorInfo);
    }

    if (biasEnabled)
    {
        if (biasTensorInfo.IsConstant())
        {
            auto biasTensor = CreateConstTensor(tfLiteBiasTensor, biasTensorInfo);

            auto biasName = GetName(armnn::LayerType::Constant, nodeIndex, "Bias");
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
        auto input = CreateConstTensor(tfLiteInputTensor, inputTensorInfo);

        auto inputName = GetName(armnn::LayerType::Constant, nodeIndex, "Input");
        armnn::IConnectableLayer* inputLayer = delegateData.m_Network->AddConstantLayer(input, inputName.c_str());
        inputLayer->GetOutputSlot(0).Connect(layer->GetInputSlot(0u));
        inputLayer->GetOutputSlot(0).SetTensorInfo(inputTensorInfo);
    }

    ARMNN_ASSERT(layer != nullptr);

    armnn::IOutputSlot& outputSlot = layer->GetOutputSlot(0);
    outputSlot.SetTensorInfo(outputTensorInfo);

    if (Connect(layer, tfLiteContext, tfLiteNode, delegateData) != kTfLiteOk)
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

TfLiteStatus VisitDepthwiseConv2dOperator(DelegateData& delegateData,
                                          TfLiteOpaqueContext* tfLiteContext,
                                          TfLiteOpaqueNode* tfLiteNode,
                                          int nodeIndex,
                                          int32_t operatorCode)
{
    auto numInputs = TfLiteOpaqueNodeNumberOfInputs(tfLiteNode);
    if (numInputs < 2)
    {
        TF_LITE_OPAQUE_MAYBE_KERNEL_LOG(
                tfLiteContext,
                "TfLiteArmnnOpaqueDelegate: Minimum number of inputs (%d != %d) in node #%d",
                2, numInputs, nodeIndex);
        return kTfLiteError;
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

    // Use input indices to get filter tensor.
    const TfLiteOpaqueTensor* tfLiteFilterTensor = TfLiteOpaqueContextGetOpaqueTensor(tfLiteContext, inputTensors[1]);
    if (!IsValid(tfLiteContext, tfLiteFilterTensor, operatorCode, nodeIndex))
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

    const armnn::TensorInfo& inputTensorInfo  = GetTensorInfoForTfLiteOpaqueTensor(tfLiteInputTensor);
    const armnn::TensorInfo& filterTensorInfo = GetTensorInfoForTfLiteOpaqueTensor(tfLiteFilterTensor);
    const armnn::TensorInfo& outputTensorInfo = GetTensorInfoForTfLiteOpaqueTensor(tfLiteOutputTensor, true);

    auto* tfLiteNodeParameters =
            reinterpret_cast<TfLiteDepthwiseConvParams*>(TfLiteOpaqueNodeGetBuiltinData(tfLiteNode));

    TfLiteFusedActivation activationType = kTfLiteActNone;
    if (tfLiteNodeParameters)
    {
        activationType = tfLiteNodeParameters->activation;
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

    armnn::TensorInfo biasTensorInfo;
    const TfLiteOpaqueTensor* tfLiteBiasTensor = nullptr;

    bool biasEnabled = IsOptionalOperandPresent(tfLiteNode, 2);
    if(biasEnabled)
    {
        // Use input indices to get bias tensor.
        tfLiteBiasTensor = TfLiteOpaqueContextGetOpaqueTensor(tfLiteContext, inputTensors[2]);
        if (!IsValid(tfLiteContext, tfLiteBiasTensor, operatorCode, nodeIndex))
        {
            return kTfLiteError;
        }
        biasTensorInfo = GetTensorInfoForTfLiteOpaqueTensor(tfLiteBiasTensor);
    }
    else
    {
        biasTensorInfo = armnn::TensorInfo(armnn::TensorShape({1}), GetDataType(tfLiteInputTensor));
    }

    armnn::DepthwiseConvolution2dDescriptor descriptor;
    descriptor.m_BiasEnabled = biasEnabled;
    descriptor.m_StrideX = NonNegative(tfLiteNodeParameters->stride_width, nodeIndex);
    descriptor.m_StrideY = NonNegative(tfLiteNodeParameters->stride_height, nodeIndex);
    descriptor.m_DataLayout = armnn::DataLayout::NHWC;
    descriptor.m_DilationX = NonNegative(tfLiteNodeParameters->dilation_width_factor, nodeIndex);
    descriptor.m_DilationY = NonNegative(tfLiteNodeParameters->dilation_height_factor, nodeIndex);

    // Assuming input is NHWC
    unsigned int inputHeight = inputTensorInfo.GetShape()[1];
    unsigned int inputWidth  = inputTensorInfo.GetShape()[2];

    // TensorflowLite weights come in the format [1, H, W, I * M]
    unsigned int filterHeight = filterTensorInfo.GetShape()[1];
    unsigned int filterWidth  = filterTensorInfo.GetShape()[2];

    // Calculate padding
    CalcPadding(inputHeight, filterHeight, descriptor.m_StrideY, descriptor.m_DilationY,
                descriptor.m_PadTop, descriptor.m_PadBottom, tfLiteNodeParameters->padding);
    CalcPadding(inputWidth, filterWidth, descriptor.m_StrideX, descriptor.m_DilationX,
                descriptor.m_PadLeft, descriptor.m_PadRight, tfLiteNodeParameters->padding);

    armnn::BackendId setBackend;
    if (!delegateData.m_Network)
    {
        bool filterIsConst = filterTensorInfo.IsConstant();

        if (!filterIsConst)
        {
            filterIsConst = WillInputBeOptimizedToConst(tfLiteContext, inputTensors[1]);
        }
        armnn::TensorInfo filterTensorInfoCopy(filterTensorInfo);
        filterTensorInfoCopy.SetConstant(filterIsConst);

        armnn::Optional<armnn::TensorInfo> optionalBiasInfoCopy(biasTensorInfo);

        if (biasEnabled)
        {
            bool biasIsConst = biasTensorInfo.IsConstant();

            if (!biasIsConst)
            {
                biasIsConst = WillInputBeOptimizedToConst(tfLiteContext, inputTensors[2]);
            }
            optionalBiasInfoCopy.value().SetConstant(biasIsConst);
        }

        bool isSupported = false;
        FORWARD_LAYER_OPAQUE_SUPPORT_FUNC("DEPTHWISE_CONV2D",
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

    auto layerName = GetName(armnn::LayerType::DepthwiseConvolution2d, nodeIndex);
    armnn::IConnectableLayer* layer = delegateData.m_Network->AddDepthwiseConvolution2dLayer(descriptor,
                                                                                             layerName.c_str());
    layer->SetBackendId(setBackend);

    if(filterTensorInfo.IsConstant())
    {
        // For depthwise the weights layout is the same as for tflite [1, H, W, I*M]. No permutation required.
        auto filter = CreateConstTensor(tfLiteFilterTensor, filterTensorInfo);

        auto filterName = GetName(armnn::LayerType::Constant, nodeIndex, "Filter");
        armnn::IConnectableLayer* weightsLayer = delegateData.m_Network->AddConstantLayer(filter, filterName.c_str());
        weightsLayer->GetOutputSlot(0).Connect(layer->GetInputSlot(1u));
        weightsLayer->GetOutputSlot(0).SetTensorInfo(filterTensorInfo);
    }

    if (biasEnabled)
    {
        if(biasTensorInfo.IsConstant())
        {
            auto biasTensor = CreateConstTensor(tfLiteBiasTensor, biasTensorInfo);

            auto biasName = GetName(armnn::LayerType::Constant, nodeIndex, "Bias");
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
        auto input = CreateConstTensor(tfLiteInputTensor, inputTensorInfo);

        auto inputName = GetName(armnn::LayerType::Constant, nodeIndex, "Input");
        armnn::IConnectableLayer* inputLayer = delegateData.m_Network->AddConstantLayer(input, inputName.c_str());
        inputLayer->GetOutputSlot(0).Connect(layer->GetInputSlot(0u));
        inputLayer->GetOutputSlot(0).SetTensorInfo(inputTensorInfo);
    }

    ARMNN_ASSERT(layer != nullptr);

    armnn::IOutputSlot& outputSlot = layer->GetOutputSlot(0);
    outputSlot.SetTensorInfo(outputTensorInfo);

    if(Connect(layer, tfLiteContext, tfLiteNode, delegateData) != kTfLiteOk)
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

TfLiteStatus VisitConv3dOperator(DelegateData& delegateData,
                                 TfLiteOpaqueContext* tfLiteContext,
                                 TfLiteOpaqueNode* tfLiteNode,
                                 int nodeIndex,
                                 int32_t operatorCode)
{
    auto numInputs = TfLiteOpaqueNodeNumberOfInputs(tfLiteNode);
    if (numInputs < 2)
    {
        TF_LITE_OPAQUE_MAYBE_KERNEL_LOG(
                tfLiteContext, "TfLiteArmnnOpaqueDelegate: Minimum number of inputs (%d != %d) in node #%d",
                2, numInputs, nodeIndex);
        return kTfLiteError;
    }
    TF_LITE_ENSURE_STATUS(ValidateNumOutputs(tfLiteContext, tfLiteNode, 1, nodeIndex));

    armnn::Convolution3dDescriptor descriptor;
    auto* params = reinterpret_cast<TfLiteConv3DParams*>(TfLiteOpaqueNodeGetBuiltinData(tfLiteNode));

    bool biasEnabled = IsOptionalOperandPresent(tfLiteNode, 2);
    descriptor.m_BiasEnabled = biasEnabled;
    descriptor.m_DataLayout = armnn::DataLayout::NDHWC;
    descriptor.m_StrideX = NonNegative(params->stride_width, nodeIndex);
    descriptor.m_StrideY = NonNegative(params->stride_height, nodeIndex);
    descriptor.m_StrideZ = NonNegative(params->stride_depth, nodeIndex);
    descriptor.m_DilationX = NonNegative(params->dilation_width_factor, nodeIndex);
    descriptor.m_DilationY = NonNegative(params->dilation_height_factor, nodeIndex);
    descriptor.m_DilationZ = NonNegative(params->dilation_depth_factor, nodeIndex);

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

    // Use input indices to get filter tensor.
    const TfLiteOpaqueTensor* tfLiteFilterTensor = TfLiteOpaqueContextGetOpaqueTensor(tfLiteContext, inputTensors[1]);
    if (!IsValid(tfLiteContext, tfLiteFilterTensor, operatorCode, nodeIndex))
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

    const armnn::TensorInfo& inputTensorInfo  = GetTensorInfoForTfLiteOpaqueTensor(tfLiteInputTensor);
    const armnn::TensorInfo& outputTensorInfo = GetTensorInfoForTfLiteOpaqueTensor(tfLiteOutputTensor, true);

    auto* tfLiteNodeParameters = reinterpret_cast<TfLiteConv3DParams*>(TfLiteOpaqueNodeGetBuiltinData(tfLiteNode));
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

    const armnn::TensorInfo& filterTensorInfo = GetTensorInfoForTfLiteOpaqueTensor(tfLiteFilterTensor);

    armnn::TensorInfo biasTensorInfo;
    const TfLiteOpaqueTensor* tfLiteBiasTensor = nullptr;

    if (biasEnabled)
    {
        // Use input indices to get bias tensor.
        tfLiteBiasTensor = TfLiteOpaqueContextGetOpaqueTensor(tfLiteContext, inputTensors[2]);
        if (!IsValid(tfLiteContext, tfLiteBiasTensor, operatorCode, nodeIndex))
        {
            return kTfLiteError;
        }
        biasTensorInfo = GetTensorInfoForTfLiteOpaqueTensor(tfLiteBiasTensor);
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
        FORWARD_LAYER_OPAQUE_SUPPORT_FUNC("CONV3D",
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

    auto layerName = GetName(armnn::LayerType::Convolution3d, nodeIndex);
    armnn::IConnectableLayer* layer =  delegateData.m_Network->AddConvolution3dLayer(descriptor, layerName.c_str());
    layer->SetBackendId(setBackend);
    ARMNN_ASSERT(layer != nullptr);

    // Add a constant layer for weights and biases if inputs are constant,
    // which are connected to the Convolution3d layer as inputs.
    if (filterTensorInfo.IsConstant())
    {
        auto filter = CreateConstTensor(tfLiteFilterTensor,
                                        filterTensorInfo);

        auto filterName = GetName(armnn::LayerType::Constant, nodeIndex, "Filter");
        armnn::IConnectableLayer* weightsLayer = delegateData.m_Network->AddConstantLayer(filter, filterName.c_str());
        ARMNN_ASSERT(weightsLayer != nullptr);

        weightsLayer->GetOutputSlot(0).Connect(layer->GetInputSlot(1u));
        weightsLayer->GetOutputSlot(0).SetTensorInfo(filterTensorInfo);
    }

    if (biasEnabled)
    {
        if (biasTensorInfo.IsConstant())
        {
            auto biasTensor = CreateConstTensor(tfLiteBiasTensor, biasTensorInfo);

            auto biasName = GetName(armnn::LayerType::Constant, nodeIndex, "Bias");
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
        auto input = CreateConstTensor(tfLiteInputTensor, inputTensorInfo);

        auto inputName = GetName(armnn::LayerType::Constant, nodeIndex, "Input");
        armnn::IConnectableLayer* inputLayer = delegateData.m_Network->AddConstantLayer(input, inputName.c_str());
        inputLayer->GetOutputSlot(0).Connect(layer->GetInputSlot(0u));
        inputLayer->GetOutputSlot(0).SetTensorInfo(inputTensorInfo);
    }

    armnn::IOutputSlot& outputSlot = layer->GetOutputSlot(0);
    outputSlot.SetTensorInfo(outputTensorInfo);

    if (Connect(layer, tfLiteContext, tfLiteNode, delegateData) != kTfLiteOk)
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
                                          TfLiteOpaqueContext* tfLiteContext,
                                          TfLiteOpaqueNode* tfLiteNode,
                                          int nodeIndex,
                                          int32_t operatorCode)
{
    armnn::TransposeConvolution2dDescriptor descriptor;
    auto* parameters = reinterpret_cast<TfLiteTransposeConvParams*>(TfLiteOpaqueNodeGetBuiltinData(tfLiteNode));
    descriptor.m_BiasEnabled = false;
    // Bias is optional.
    auto numInputs = TfLiteOpaqueNodeNumberOfInputs(tfLiteNode);
    switch (numInputs)
    {
        case 3:
            descriptor.m_BiasEnabled = false;
            break;
        case 4:
            descriptor.m_BiasEnabled = true;
            break;
        default:
            TF_LITE_OPAQUE_MAYBE_KERNEL_LOG(
                tfLiteContext,
                "TfLiteArmnnOpaqueDelegate: TransposeConv2d num inputs specified, %d, is out of range 3-4 on node #%d",
                numInputs, nodeIndex);
            return kTfLiteError;
    }

    TF_LITE_ENSURE_STATUS(ValidateNumOutputs(tfLiteContext, tfLiteNode, 1, nodeIndex));
    descriptor.m_StrideX = NonNegative(parameters->stride_width, nodeIndex);
    descriptor.m_StrideY = NonNegative(parameters->stride_height, nodeIndex);
    descriptor.m_DataLayout = armnn::DataLayout::NHWC;
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

    const TfLiteOpaqueTensor* tfLiteOutputShapeTensor = TfLiteOpaqueContextGetOpaqueTensor(tfLiteContext,
                                                                                           inputTensors[0]);
    if (!IsValid(tfLiteContext, tfLiteOutputShapeTensor, operatorCode, nodeIndex))
    {
        return kTfLiteError;
    }

    const TfLiteOpaqueTensor* tfLiteInputTensor = TfLiteOpaqueContextGetOpaqueTensor(tfLiteContext, inputTensors[2]);
    if (!IsValid(tfLiteContext, tfLiteInputTensor, operatorCode, nodeIndex))
    {
        return kTfLiteError;
    }

    const TfLiteOpaqueTensor* tfLiteFilterTensor = TfLiteOpaqueContextGetOpaqueTensor(tfLiteContext, inputTensors[1]);
    if (!IsValid(tfLiteContext, tfLiteFilterTensor, operatorCode, nodeIndex))
    {
        return kTfLiteError;
    }

    TfLiteOpaqueTensor* tfLiteBiasTensor = nullptr;
    if (descriptor.m_BiasEnabled)
    {
        tfLiteBiasTensor = TfLiteOpaqueContextGetOpaqueTensor(tfLiteContext, inputTensors[3]);
        if (!IsValid(tfLiteContext, tfLiteFilterTensor, operatorCode, nodeIndex))
        {
            return kTfLiteError;
        }
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

    const armnn::TensorInfo& inputTensorInfo  = GetTensorInfoForTfLiteOpaqueTensor(tfLiteInputTensor);
    const armnn::TensorInfo& outputTensorInfo = GetTensorInfoForTfLiteOpaqueTensor(tfLiteOutputTensor, true);
    const armnn::TensorInfo& filterTensorInfo = GetTensorInfoForTfLiteOpaqueTensor(tfLiteFilterTensor);

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
    if (IsConstantTensor(tfLiteOutputShapeTensor))
    {
        const armnn::TensorInfo outputShapeTensorInfo = GetTensorInfoForTfLiteOpaqueTensor(tfLiteOutputShapeTensor);
        std::vector<int32_t> outputShape(outputShapeTensorInfo.GetNumElements());
        if (outputShapeTensorInfo.GetDataType() == armnn::DataType::Signed32)
        {
            for(unsigned int i=0; i < outputShapeTensorInfo.GetNumElements(); ++i)
            {
                outputShape[i] = static_cast<int32_t*>(TfLiteOpaqueTensorData(tfLiteOutputShapeTensor))[i];
            }
        }

        if (outputShapeTensorInfo.GetDataType() == armnn::DataType::QAsymmU8)
        {
            for(unsigned int i=0; i < outputShapeTensorInfo.GetNumElements(); ++i)
            {
                outputShape[i] = static_cast<uint8_t*>(TfLiteOpaqueTensorData(tfLiteOutputShapeTensor))[i];
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
    auto filterTensor = CreateConstTensor(tfLiteFilterTensor,
                                          filterTensorInfo);
    armnn::Optional<armnn::TensorInfo> optionalBiasInfoCopy;
    if (descriptor.m_BiasEnabled)
    {
        optionalBiasInfoCopy = armnn::Optional(GetTensorInfoForTfLiteOpaqueTensor(tfLiteBiasTensor));
        bool biasIsConst = optionalBiasInfoCopy.value().IsConstant();
        if (!biasIsConst)
        {
            biasIsConst = WillInputBeOptimizedToConst(tfLiteContext, inputTensors[3]);
        }
        // At this point if the bias isn't going to be a const tensor we're in trouble. Calls to
        // AddTransposeConvolution2dLayer expect only optional ConstTensor.
        if(!biasIsConst)
        {
            TF_LITE_OPAQUE_MAYBE_KERNEL_LOG(
                tfLiteContext,
                "TfLiteArmnnOpaqueDelegate: Non constant bias found in TransposeConv2d, node #%d: ",
                nodeIndex);
            return kTfLiteError;
        }
        optionalBiasInfoCopy.value().SetConstant(biasIsConst);
    }

    armnn::BackendId setBackend;
    if (!delegateData.m_Network)
    {
        bool isSupported = false;
        FORWARD_LAYER_OPAQUE_SUPPORT_FUNC("TRANSPOSE_CONV2D",
                                          tfLiteContext,
                                          IsTransposeConvolution2dSupported,
                                          delegateData.m_Backends,
                                          isSupported,
                                          setBackend,
                                          inputTensorInfo,
                                          outputTensorInfo,
                                          descriptor,
                                          filterTensorInfo,
                                          optionalBiasInfoCopy);
        return isSupported ? kTfLiteOk : kTfLiteError;
    }

    auto layerName = GetName(armnn::LayerType::TransposeConvolution2d, nodeIndex);

    armnn::IConnectableLayer* layer;
    armnn::Optional<armnn::ConstTensor> optionalBiases;
    if (descriptor.m_BiasEnabled)
    {
        auto biasTensor = CreateConstTensor(tfLiteBiasTensor,
                                            GetTensorInfoForTfLiteOpaqueTensor(tfLiteBiasTensor));
        optionalBiases = armnn::MakeOptional<armnn::ConstTensor>(biasTensor);
    }
    layer = delegateData.m_Network->AddTransposeConvolution2dLayer(descriptor,
                                                                   filterTensor,
                                                                   optionalBiases,
                                                                   layerName.c_str());
    layer->SetBackendId(setBackend);
    ARMNN_ASSERT(layer != nullptr);

    // The data input can be constant, so we must check that this is allocated to an input slot
    if(inputTensorInfo.IsConstant())
    {
        auto input = CreateConstTensor(tfLiteInputTensor, inputTensorInfo);

        auto inputName = GetName(armnn::LayerType::Constant, nodeIndex, "Input");
        armnn::IConnectableLayer *inputLayer = delegateData.m_Network->AddConstantLayer(input, inputName.c_str());
        inputLayer->GetOutputSlot(0).Connect(layer->GetInputSlot(0u));
        inputLayer->GetOutputSlot(0).SetTensorInfo(inputTensorInfo);
    }

    armnn::IOutputSlot& outputSlot = layer->GetOutputSlot(0);
    outputSlot.SetTensorInfo(outputTensorInfo);


    // Connect
    if (delegateData.m_OutputSlotForNode[static_cast<unsigned int>(inputTensors[2])] != nullptr)
    {
        delegateData.m_OutputSlotForNode[static_cast<unsigned int>(inputTensors[2])]->
                Connect(layer->GetInputSlot(0));
    }

    if (Connect(layer, tfLiteContext, tfLiteNode, delegateData) != kTfLiteOk)
    {
        return kTfLiteError;
    }

    if (activationType != kTfLiteActNone)
    {
        return FusedActivation(tfLiteContext, tfLiteNode, activationType, layer, 0, delegateData, nodeIndex);
    }

    return kTfLiteOk;
}

TfLiteStatus VisitConvolutionOperator(DelegateData& delegateData,
                                      TfLiteOpaqueContext* tfLiteContext,
                                      TfLiteOpaqueNode* tfLiteNode,
                                      int nodeIndex,
                                      int32_t operatorCode)
{
    switch(operatorCode)
    {
        case kTfLiteBuiltinConv2d:
            return VisitConv2dOperator(delegateData, tfLiteContext, tfLiteNode, nodeIndex, operatorCode);
        case kTfLiteBuiltinConv3d:
            return VisitConv3dOperator(delegateData, tfLiteContext, tfLiteNode, nodeIndex, operatorCode);
        case kTfLiteBuiltinDepthwiseConv2d:
            return VisitDepthwiseConv2dOperator(delegateData, tfLiteContext, tfLiteNode, nodeIndex, operatorCode);
        case kTfLiteBuiltinTransposeConv:
            return VisitTransposeConv2dOperator(delegateData, tfLiteContext, tfLiteNode, nodeIndex, operatorCode);
        default:
            return kTfLiteError;
    }
}

}