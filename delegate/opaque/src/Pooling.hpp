//
// Copyright © 2023 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include <OpaqueDelegateUtils.hpp>
#include <SharedFunctions.hpp>

#include <flatbuffers/flexbuffers.h>

namespace armnnOpaqueDelegate
{

TfLiteStatus VisitPooling2dOperator(DelegateData& delegateData,
                                    TfLiteOpaqueContext* tfLiteContext,
                                    TfLiteOpaqueNode* tfLiteNode,
                                    int nodeIndex,
                                    int32_t tfLitePoolingOperatorCode)
{
    TF_LITE_ENSURE_STATUS(ValidateNumInputs(tfLiteContext, tfLiteNode, 1, nodeIndex));
    TF_LITE_ENSURE_STATUS(ValidateNumOutputs(tfLiteContext, tfLiteNode, 1, nodeIndex));

    // Gather input indices and use to get input tensors.
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
    if (!IsValid(tfLiteContext, tfLiteInputTensor, tfLitePoolingOperatorCode, nodeIndex))
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
    if (!IsValid(tfLiteContext, tfLiteOutputTensor, tfLitePoolingOperatorCode, nodeIndex))
    {
        return kTfLiteError;
    }

    const armnn::TensorInfo& inputTensorInfo = GetTensorInfoForTfLiteOpaqueTensor(tfLiteInputTensor);
    const armnn::TensorInfo& outputTensorInfo = GetTensorInfoForTfLiteOpaqueTensor(tfLiteOutputTensor, true);

    auto* tfLiteNodeParameters = reinterpret_cast<TfLitePoolParams*>(TfLiteOpaqueNodeGetBuiltinData(tfLiteNode));
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

    armnn::PoolingAlgorithm poolingAlgorithm;
    switch(tfLitePoolingOperatorCode)
    {
        case kTfLiteBuiltinAveragePool2d:
            poolingAlgorithm = armnn::PoolingAlgorithm::Average;
            break;
        case kTfLiteBuiltinL2Pool2d:
            poolingAlgorithm = armnn::PoolingAlgorithm::L2;
            break;
        case kTfLiteBuiltinMaxPool2d:
            poolingAlgorithm = armnn::PoolingAlgorithm::Max;
            break;
        default:
            return kTfLiteError;
    }

    armnn::Pooling2dDescriptor descriptor;
    descriptor.m_PoolType = poolingAlgorithm;

    descriptor.m_PoolWidth = tfLiteNodeParameters->filter_width;
    descriptor.m_PoolHeight = tfLiteNodeParameters->filter_height;
    descriptor.m_StrideX = tfLiteNodeParameters->stride_width;
    descriptor.m_StrideY = tfLiteNodeParameters->stride_height;
    descriptor.m_DataLayout = armnn::DataLayout::NHWC;

    unsigned int inputHeight = inputTensorInfo.GetShape()[1];
    unsigned int inputWidth  = inputTensorInfo.GetShape()[2];

    CalcPadding(inputHeight, descriptor.m_PoolHeight, descriptor.m_StrideY, 1u,
                descriptor.m_PadTop, descriptor.m_PadBottom, tfLiteNodeParameters->padding);
    CalcPadding(inputWidth, descriptor.m_PoolWidth, descriptor.m_StrideX, 1u,
                descriptor.m_PadLeft, descriptor.m_PadRight, tfLiteNodeParameters->padding);

    bool isSupported = false;
    armnn::BackendId setBackend;
    auto validateFunc = [&](const armnn::TensorInfo& outputTensorInfo, bool& isSupported)
    {
        FORWARD_LAYER_OPAQUE_SUPPORT_FUNC("POOLING_2D",
                                          tfLiteContext,
                                          IsPooling2dSupported,
                                          delegateData.m_Backends,
                                          isSupported,
                                          setBackend,
                                          inputTensorInfo,
                                          outputTensorInfo,
                                          descriptor);
    };

    if (!delegateData.m_Network)
    {
        validateFunc(outputTensorInfo, isSupported);
        return isSupported ? kTfLiteOk : kTfLiteError;
    }

    auto layerName = GetName(armnn::LayerType::Pooling2d, nodeIndex);
    armnn::IConnectableLayer* poolingLayer = delegateData.m_Network->AddPooling2dLayer(descriptor, layerName.c_str());
    poolingLayer->SetBackendId(setBackend);
    ARMNN_ASSERT(poolingLayer != nullptr);

    armnn::IOutputSlot& outputSlot = poolingLayer->GetOutputSlot(0);
    outputSlot.SetTensorInfo(outputTensorInfo);

    // try to connect the Constant Inputs if there are any
    if (ProcessInputs(poolingLayer, delegateData, tfLiteContext, tfLiteNode, nodeIndex) != kTfLiteOk)
    {
        return kTfLiteError;
    }

    if (Connect(poolingLayer, tfLiteContext, tfLiteNode, delegateData) != kTfLiteOk)
    {
        return kTfLiteError;
    }

    // Check and create activation
    return FusedActivation(tfLiteContext, tfLiteNode, activationType, poolingLayer, 0, delegateData, nodeIndex);
}

TfLiteStatus VisitPooling3dOperator(DelegateData& delegateData,
                                    TfLiteOpaqueContext* tfLiteContext,
                                    TfLiteOpaqueNode* tfLiteNode,
                                    int nodeIndex,
                                    std::string customOperatorName)
{
    TF_LITE_ENSURE_STATUS(ValidateNumInputs(tfLiteContext, tfLiteNode, 1, nodeIndex));
    TF_LITE_ENSURE_STATUS(ValidateNumOutputs(tfLiteContext, tfLiteNode, 1, nodeIndex));

    // Gather input indices and use to get input tensors.
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
    if (!IsValid(tfLiteContext, tfLiteInputTensor, kTfLiteBuiltinCustom, nodeIndex))
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
    if (!IsValid(tfLiteContext, tfLiteOutputTensor, kTfLiteBuiltinCustom, nodeIndex))
    {
        return kTfLiteError;
    }

    // Set the input and output info
    const armnn::TensorInfo& inputTensorInfo = GetTensorInfoForTfLiteOpaqueTensor(tfLiteInputTensor);
    const armnn::TensorInfo& outputTensorInfo = GetTensorInfoForTfLiteOpaqueTensor(tfLiteOutputTensor, true);

    // Custom Operators are defined by the name string associated to the operator. Use this to determine
    // which pooling algorithm to create the armnn operator with. L2 Pooling3D is unsupported in TfLite.
    armnn::PoolingAlgorithm poolingAlgorithm;
    if (customOperatorName == "MaxPool3D")
    {
        poolingAlgorithm = armnn::PoolingAlgorithm::Max;
    }
    else if (customOperatorName == "AveragePool3D")
    {
        poolingAlgorithm = armnn::PoolingAlgorithm::Average;
    }
    else
    {
        return kTfLiteError;
    }
    // Create the armnn pool3d descriptor and set the algorithm parsed above.
    armnn::Pooling3dDescriptor descriptor;
    descriptor.m_PoolType = poolingAlgorithm;

    // custom_initial_data and custom_initial_data_size are void* variables defined in the tflite registration
    // used to access the custom option buffer for the operator.
    const void* customData = nullptr;
    int customDataSize = 0;
    if (TfLiteOpaqueNodeGetCustomInitialData(tfLiteNode, &customData, &customDataSize) != kTfLiteOk)
    {
        TF_LITE_OPAQUE_MAYBE_KERNEL_LOG(
                tfLiteContext,
                "TfLiteArmnnOpaqueDelegate: Unable to initialise initial custom data from node #%d: ",
                nodeIndex);
        return kTfLiteError;
    }

    // Reinterpret the void* to a byte buffer to access the options data in the flexbuffers map.
    const flexbuffers::Map& m = flexbuffers::GetRoot(reinterpret_cast<const uint8_t*>(customData),
                                                     customDataSize).AsMap();
    // poolDims is a vector of [ 1, Depth, Height, Width, 1 ]
    const auto poolDims = m["ksize"].AsTypedVector();
    descriptor.m_PoolWidth = poolDims[3].AsInt32();
    descriptor.m_PoolHeight = poolDims[2].AsInt32();
    descriptor.m_PoolDepth = poolDims[1].AsInt32();

    // strideDimes is a vector of [ 1, Z, Y, X, 1]
    const auto strideDims = m["strides"].AsTypedVector();
    descriptor.m_StrideX = strideDims[3].AsInt32();
    descriptor.m_StrideY = strideDims[2].AsInt32();
    descriptor.m_StrideZ = strideDims[1].AsInt32();
    descriptor.m_DataLayout = armnn::DataLayout::NDHWC;

    unsigned int inputDepth = inputTensorInfo.GetShape()[1];
    unsigned int inputHeight = inputTensorInfo.GetShape()[2];
    unsigned int inputWidth = inputTensorInfo.GetShape()[3];

    // CalcPadding expects a TfLitePadding type. Parse flexbuffers to extract padding string and create TfLitePadding.
    std::string paddingStr = m["padding"].AsString().str();
    TfLitePadding padding;
    if (paddingStr == "VALID")
    {
        padding = kTfLitePaddingValid;
    }
    else if (paddingStr == "SAME")
    {
        padding = kTfLitePaddingSame;
    }
    else
    {
        padding = kTfLitePaddingUnknown;
    }
    // Calculates padding for each pooling dimension separately
    CalcPadding(inputHeight, descriptor.m_PoolHeight, descriptor.m_StrideY, 1u,
                descriptor.m_PadTop, descriptor.m_PadBottom, padding);
    CalcPadding(inputWidth, descriptor.m_PoolWidth, descriptor.m_StrideX, 1u,
                descriptor.m_PadLeft, descriptor.m_PadRight, padding);
    CalcPadding(inputDepth, descriptor.m_PoolDepth, descriptor.m_StrideZ, 1u,
                descriptor.m_PadFront, descriptor.m_PadBack, padding);


    // Check activation by parsing the string from the flexbuffer map
    std::string activationTypeStr = m["activation"].AsString().str();
    TfLiteFusedActivation activationType = kTfLiteActNone;

    if (activationTypeStr == "kTfLiteActRelu")
    {
        activationType = kTfLiteActRelu;
    }
    else if (activationTypeStr == "kTfLiteActReluN1To1")
    {
        activationType = kTfLiteActReluN1To1;
    }
    else if (activationTypeStr == "kTfLiteActRelu6")
    {
        activationType = kTfLiteActRelu6;
    }
    else if (activationTypeStr == "kTfLiteActTanh")
    {
        activationType = kTfLiteActTanh;
    }
    else if (activationTypeStr == "kTfLiteActSignBit")
    {
        activationType = kTfLiteActSignBit;
    }
    else if (activationTypeStr == "kTfLiteActSigmoid")
    {
        activationType = kTfLiteActSigmoid;
    }
    else
    {
        activationType = kTfLiteActNone;
    }

    TfLiteStatus activationStatus = ValidateFusedActivationOperator(delegateData,
                                                                    tfLiteContext,
                                                                    outputTensorInfo,
                                                                    outputTensorInfo,
                                                                    activationType);
    if(activationStatus != kTfLiteOk)
    {
        return kTfLiteError;
    }

    // Validate the output info.
    bool isSupported = false;
    armnn::BackendId setBackend;
    auto validateFunc = [&](const armnn::TensorInfo& outputTensorInfo, bool& isSupported)
    {
        FORWARD_LAYER_OPAQUE_SUPPORT_FUNC("POOLING_3D",
                                          tfLiteContext,
                                          IsPooling3dSupported,
                                          delegateData.m_Backends,
                                          isSupported,
                                          setBackend,
                                          inputTensorInfo,
                                          outputTensorInfo,
                                          descriptor);
    };

    if (!delegateData.m_Network)
    {
        validateFunc(outputTensorInfo, isSupported);
        return isSupported ? kTfLiteOk : kTfLiteError;
    }

    // Create the Layer
    auto layerName = GetName(armnn::LayerType::Pooling3d, nodeIndex);
    armnn::IConnectableLayer* poolingLayer = delegateData.m_Network->AddPooling3dLayer(descriptor, layerName.c_str());
    poolingLayer->SetBackendId(setBackend);
    ARMNN_ASSERT(poolingLayer != nullptr);

    // Create and set output slots
    armnn::IOutputSlot& outputSlot = poolingLayer->GetOutputSlot(0);
    outputSlot.SetTensorInfo(outputTensorInfo);

    // try to connect the Constant Inputs if there are any
    if (ProcessInputs(poolingLayer, delegateData, tfLiteContext, tfLiteNode, nodeIndex) != kTfLiteOk)
    {
        return kTfLiteError;
    }

    if (Connect(poolingLayer, tfLiteContext, tfLiteNode, delegateData) != kTfLiteOk)
    {
        return kTfLiteError;
    }

    return FusedActivation(tfLiteContext, tfLiteNode, activationType, poolingLayer, 0, delegateData, nodeIndex);
}

} // namespace armnnOpaqueDelegate
