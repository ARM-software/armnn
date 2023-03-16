//
// Copyright © 2022-2023 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include <ClassicDelegateUtils.hpp>

#include <tensorflow/lite/builtin_ops.h>
#include <tensorflow/lite/c/builtin_op_data.h>
#include <tensorflow/lite/c/common.h>
#include <tensorflow/lite/minimal_logging.h>
#include <flatbuffers/flexbuffers.h>

namespace armnnDelegate
{

TfLiteStatus VisitPooling2dOperator(DelegateData& delegateData,
                                    TfLiteContext* tfLiteContext,
                                    TfLiteNode* tfLiteNode,
                                    int nodeIndex,
                                    int32_t tfLitePoolingOperatorCode)
{
    TF_LITE_ENSURE_STATUS(ValidateNumInputs(tfLiteContext, tfLiteNode, 1, nodeIndex));
    TF_LITE_ENSURE_STATUS(ValidateNumOutputs(tfLiteContext, tfLiteNode, 1, nodeIndex));

    const TfLiteTensor* tfLiteTensors = tfLiteContext->tensors;
    const TfLiteTensor& tfLiteInputTensor = tfLiteTensors[tfLiteNode->inputs->data[0]];
    if (IsDynamicTensor(tfLiteInputTensor))
    {
        TF_LITE_MAYBE_KERNEL_LOG(
            tfLiteContext,
            "TfLiteArmnnDelegate: Dynamic input tensors are not supported in operator #%d node #%d: ",
            tfLitePoolingOperatorCode, nodeIndex);
        return kTfLiteError;
    }

    const TfLiteTensor& tfLiteOutputTensor = tfLiteTensors[tfLiteNode->outputs->data[0]];
    if (IsDynamicTensor(tfLiteOutputTensor))
    {
        TF_LITE_MAYBE_KERNEL_LOG(
            tfLiteContext,
            "TfLiteArmnnDelegate: Dynamic output tensors are not supported in operator #%d node #%d: ",
            tfLitePoolingOperatorCode, nodeIndex);
        return kTfLiteError;
    }

    const armnn::TensorInfo& inputTensorInfo = GetTensorInfoForTfLiteTensor(tfLiteInputTensor);
    const armnn::TensorInfo& outputTensorInfo = GetTensorInfoForTfLiteTensor(tfLiteOutputTensor, true);

    auto* tfLiteNodeParameters = reinterpret_cast<TfLitePoolParams*>(tfLiteNode->builtin_data);
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
        FORWARD_LAYER_SUPPORT_FUNC("POOLING_2D",
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

    armnn::IConnectableLayer* poolingLayer = delegateData.m_Network->AddPooling2dLayer(descriptor);
    poolingLayer->SetBackendId(setBackend);
    ARMNN_ASSERT(poolingLayer != nullptr);

    armnn::IOutputSlot& outputSlot = poolingLayer->GetOutputSlot(0);
    outputSlot.SetTensorInfo(outputTensorInfo);

    // try to connect the Constant Inputs if there are any
    if(ProcessInputs(poolingLayer,delegateData, tfLiteContext, tfLiteNode) != kTfLiteOk )
    {
        return kTfLiteError;
    }

    if(Connect(poolingLayer, tfLiteNode, delegateData) != kTfLiteOk)
    {
        return kTfLiteError;
    }

    // Check and create activation
    return FusedActivation(tfLiteContext, tfLiteNode, activationType, poolingLayer, 0, delegateData);
}

TfLiteStatus VisitPooling3dOperator(DelegateData& delegateData,
                                    TfLiteContext* tfLiteContext,
                                    TfLiteNode* tfLiteNode,
                                    int nodeIndex,
                                    std::string customOperatorName)
{
    TF_LITE_ENSURE_STATUS(ValidateNumInputs(tfLiteContext, tfLiteNode, 1, nodeIndex));
    TF_LITE_ENSURE_STATUS(ValidateNumOutputs(tfLiteContext, tfLiteNode, 1, nodeIndex));

    const TfLiteTensor* tfLiteTensors = tfLiteContext->tensors;
    const TfLiteTensor& tfLiteInputTensor = tfLiteTensors[tfLiteNode->inputs->data[0]];
    if (IsDynamicTensor(tfLiteInputTensor))
    {
        TF_LITE_MAYBE_KERNEL_LOG(
            tfLiteContext,
            "TfLiteArmnnDelegate: Dynamic input tensors are not supported in operator #%d node #%d: ",
            customOperatorName.c_str(), nodeIndex);
        return kTfLiteError;
    }

    const TfLiteTensor& tfLiteOutputTensor = tfLiteTensors[tfLiteNode->outputs->data[0]];
    if (IsDynamicTensor(tfLiteOutputTensor))
    {
        TF_LITE_MAYBE_KERNEL_LOG(
            tfLiteContext,
            "TfLiteArmnnDelegate: Dynamic output tensors are not supported in operator #%d node #%d: ",
            customOperatorName.c_str(), nodeIndex);
        return kTfLiteError;
    }
    // Set the input and output info
    const armnn::TensorInfo& inputTensorInfo = GetTensorInfoForTfLiteTensor(tfLiteInputTensor);
    const armnn::TensorInfo& outputTensorInfo = GetTensorInfoForTfLiteTensor(tfLiteOutputTensor, true);

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
    auto custom_data = tfLiteNode->custom_initial_data;
    auto custom_data_size = tfLiteNode->custom_initial_data_size;
    // Reinterpret the void* to a byte buffer to access the options data in the flexbuffers map.
    const flexbuffers::Map& m = flexbuffers::GetRoot(reinterpret_cast<const uint8_t*>(custom_data),
                                                     custom_data_size).AsMap();
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

    TfLiteStatus activationStatus = ValidateFusedActivationOperator(delegateData, tfLiteContext, outputTensorInfo,
                                                                    outputTensorInfo, activationType);
    if(activationStatus != kTfLiteOk)
    {
        return kTfLiteError;
    }


    // Validate the output info.
    bool isSupported = false;
    armnn::BackendId setBackend;
    auto validateFunc = [&](const armnn::TensorInfo& outputTensorInfo, bool& isSupported) {
        FORWARD_LAYER_SUPPORT_FUNC("POOLING_3D",
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
    armnn::IConnectableLayer* poolingLayer = delegateData.m_Network->AddPooling3dLayer(descriptor);
    poolingLayer->SetBackendId(setBackend);
    ARMNN_ASSERT(poolingLayer != nullptr);

    // Create and set output slots
    armnn::IOutputSlot& outputSlot = poolingLayer->GetOutputSlot(0);
    outputSlot.SetTensorInfo(outputTensorInfo);

    // try to connect the Constant Inputs if there are any
    if(ProcessInputs(poolingLayer,delegateData, tfLiteContext, tfLiteNode) != kTfLiteOk )
    {
        return kTfLiteError;
    }

    if(Connect(poolingLayer, tfLiteNode, delegateData) != kTfLiteOk)
    {
        return kTfLiteError;
    }

    return FusedActivation(tfLiteContext, tfLiteNode, activationType, poolingLayer, 0, delegateData);
}

} // namespace armnnDelegate
