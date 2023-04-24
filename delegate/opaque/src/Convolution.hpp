//
// Copyright © 2023 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include <OpaqueDelegateUtils.hpp>
#include <SharedFunctions.hpp>

#include <tensorflow/lite/builtin_ops.h>
#include <tensorflow/lite/c/builtin_op_data.h>
#include <tensorflow/lite/c/common.h>
#include <tensorflow/lite/minimal_logging.h>

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
    if (IsDynamicTensor(tfLiteInputTensor))
    {
        TF_LITE_OPAQUE_MAYBE_KERNEL_LOG(
                tfLiteContext,
                "TfLiteArmnnOpaqueDelegate: Dynamic input tensors are not supported in operator #%d node #%d: ",
                operatorCode, nodeIndex);
        return kTfLiteError;
    }

    // Use input indices to get filter tensor.
    const TfLiteOpaqueTensor* tfLiteFilterTensor = TfLiteOpaqueContextGetOpaqueTensor(tfLiteContext, inputTensors[1]);
    if(!IsValid(tfLiteFilterTensor))
    {
        TF_LITE_OPAQUE_MAYBE_KERNEL_LOG(
                tfLiteContext,
                "TfLiteArmnnOpaqueDelegate: Invalid filter tensor in operator #%d node #%d: ",
                operatorCode, nodeIndex);
        return kTfLiteError;
    }
    if (IsDynamicTensor(tfLiteFilterTensor))
    {
        TF_LITE_OPAQUE_MAYBE_KERNEL_LOG(
                tfLiteContext,
                "TfLiteArmnnOpaqueDelegate: Dynamic filter tensors are not supported in node #%d: ",
                nodeIndex);
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
    if (IsDynamicTensor(tfLiteOutputTensor))
    {
        TF_LITE_OPAQUE_MAYBE_KERNEL_LOG(
                tfLiteContext,
                "TfLiteArmnnOpaqueDelegate: Dynamic output tensors are not supported in operator #%d node #%d: ",
                operatorCode, nodeIndex);
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
        if(!IsValid(tfLiteBiasTensor))
        {
            TF_LITE_OPAQUE_MAYBE_KERNEL_LOG(
                    tfLiteContext,
                    "TfLiteArmnnOpaqueDelegate: Invalid bias tensor in operator #%d node #%d: ",
                    operatorCode, nodeIndex);
            return kTfLiteError;
        }
        if (IsDynamicTensor(tfLiteBiasTensor))
        {
            TF_LITE_OPAQUE_MAYBE_KERNEL_LOG(
                    tfLiteContext,
                    "TfLiteArmnnOpaqueDelegate: Dynamic bias tensors are not supported in node #%d: ",
                    nodeIndex);
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

    // Calculate padding
    CalcPadding(inputHeight, filterHeight, descriptor.m_StrideY, descriptor.m_DilationY,
                descriptor.m_PadTop, descriptor.m_PadBottom, tfLiteNodeParameters->padding);
    CalcPadding(inputWidth, filterWidth, descriptor.m_StrideX, descriptor.m_DilationX,
                descriptor.m_PadLeft, descriptor.m_PadRight, tfLiteNodeParameters->padding);

    armnn::BackendId setBackend;
    if (!delegateData.m_Network)
    {
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
                                          filterTensorInfo,
                                          optionalBiasInfo);
        return isSupported ? kTfLiteOk : kTfLiteError;
    }

    // Set up filter and biases
    armnn::IConnectableLayer* layer = delegateData.m_Network->AddConvolution2dLayer(descriptor);
    layer->SetBackendId(setBackend);

    if(filterTensorInfo.IsConstant())
    {
        auto filter = CreateConstTensor(tfLiteFilterTensor, filterTensorInfo);

        armnn::IConnectableLayer* weightsLayer = delegateData.m_Network->AddConstantLayer(filter);
        weightsLayer->GetOutputSlot(0).Connect(layer->GetInputSlot(1u));
        weightsLayer->GetOutputSlot(0).SetTensorInfo(filterTensorInfo);
    }

    if (biasEnabled)
    {
        if(biasTensorInfo.IsConstant())
        {
            auto biasTensor = CreateConstTensor(tfLiteBiasTensor, biasTensorInfo);
            armnn::IConnectableLayer* biasLayer = delegateData.m_Network->AddConstantLayer(biasTensor);
            ARMNN_ASSERT(biasLayer != nullptr);
            biasLayer->GetOutputSlot(0).Connect(layer->GetInputSlot(2u));
            biasLayer->GetOutputSlot(0).SetTensorInfo(biasTensorInfo);
        }
    }

    // The data input can also be constant, so we must check that this is also allocated to an input slot
    if(inputTensorInfo.IsConstant())
    {
        auto input = CreateConstTensor(tfLiteInputTensor, inputTensorInfo);

        armnn::IConnectableLayer* inputLayer = delegateData.m_Network->AddConstantLayer(input);
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

    // Check and Create activation
    return FusedActivation(tfLiteContext, tfLiteNode, activationType, layer, 0, delegateData);
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
    if (IsDynamicTensor(tfLiteInputTensor))
    {
        TF_LITE_OPAQUE_MAYBE_KERNEL_LOG(
                tfLiteContext,
                "TfLiteArmnnOpaqueDelegate: Dynamic input tensors are not supported in operator #%d node #%d: ",
                operatorCode, nodeIndex);
        return kTfLiteError;
    }

    // Use input indices to get filter tensor.
    const TfLiteOpaqueTensor* tfLiteFilterTensor = TfLiteOpaqueContextGetOpaqueTensor(tfLiteContext, inputTensors[1]);
    if(!IsValid(tfLiteFilterTensor))
    {
        TF_LITE_OPAQUE_MAYBE_KERNEL_LOG(
                tfLiteContext,
                "TfLiteArmnnOpaqueDelegate: Invalid filter tensor in operator #%d node #%d: ",
                operatorCode, nodeIndex);
        return kTfLiteError;
    }
    if (IsDynamicTensor(tfLiteFilterTensor))
    {
        TF_LITE_OPAQUE_MAYBE_KERNEL_LOG(
                tfLiteContext,
                "TfLiteArmnnOpaqueDelegate: Dynamic filter tensors are not supported in node #%d: ",
                nodeIndex);
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
    if (IsDynamicTensor(tfLiteOutputTensor))
    {
        TF_LITE_OPAQUE_MAYBE_KERNEL_LOG(
                tfLiteContext,
                "TfLiteArmnnOpaqueDelegate: Dynamic output tensors are not supported in operator #%d node #%d: ",
                operatorCode, nodeIndex);
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
        if(!IsValid(tfLiteBiasTensor))
        {
            TF_LITE_OPAQUE_MAYBE_KERNEL_LOG(
                    tfLiteContext,
                    "TfLiteArmnnOpaqueDelegate: Invalid bias tensor in operator #%d node #%d: ",
                    operatorCode, nodeIndex);
            return kTfLiteError;
        }
        if (IsDynamicTensor(tfLiteBiasTensor))
        {
            TF_LITE_OPAQUE_MAYBE_KERNEL_LOG(
                    tfLiteContext,
                    "TfLiteArmnnOpaqueDelegate: Dynamic bias tensors are not supported in node #%d: ",
                    nodeIndex);
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
                                          filterTensorInfo,
                                          armnn::Optional<armnn::TensorInfo>(biasTensorInfo));
        return isSupported ? kTfLiteOk : kTfLiteError;
    }

    armnn::IConnectableLayer* layer = delegateData.m_Network->AddDepthwiseConvolution2dLayer(descriptor);
    layer->SetBackendId(setBackend);

    if(filterTensorInfo.IsConstant())
    {
        // For depthwise the weights layout is the same as for tflite [1, H, W, I*M]. No permutation required.
        auto filter = CreateConstTensor(tfLiteFilterTensor, filterTensorInfo);

        armnn::IConnectableLayer* weightsLayer = delegateData.m_Network->AddConstantLayer(filter);
        weightsLayer->GetOutputSlot(0).Connect(layer->GetInputSlot(1u));
        weightsLayer->GetOutputSlot(0).SetTensorInfo(filterTensorInfo);
    }

    if (biasEnabled)
    {
        if(biasTensorInfo.IsConstant())
        {
            auto biasTensor = CreateConstTensor(tfLiteBiasTensor, biasTensorInfo);

            armnn::IConnectableLayer* biasLayer = delegateData.m_Network->AddConstantLayer(biasTensor);
            ARMNN_ASSERT(biasLayer != nullptr);
            biasLayer->GetOutputSlot(0).Connect(layer->GetInputSlot(2u));
            biasLayer->GetOutputSlot(0).SetTensorInfo(biasTensorInfo);
        }
    }

    // The data input can also be constant, so we must check that this is also allocated to an input slot
    if(inputTensorInfo.IsConstant())
    {
        auto input = CreateConstTensor(tfLiteInputTensor, inputTensorInfo);

        armnn::IConnectableLayer* inputLayer = delegateData.m_Network->AddConstantLayer(input);
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
    return FusedActivation(tfLiteContext, tfLiteNode, activationType, layer, 0, delegateData);
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
        case kTfLiteBuiltinDepthwiseConv2d:
            return VisitDepthwiseConv2dOperator(delegateData, tfLiteContext, tfLiteNode, nodeIndex, operatorCode);
        default:
            return kTfLiteError;
    }
}

}