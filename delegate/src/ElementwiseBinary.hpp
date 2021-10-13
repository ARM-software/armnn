//
// Copyright © 2020 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include "DelegateUtils.hpp"
#include "MultiLayerFacade.hpp"
#include "SharedFunctions.hpp"

#include <tensorflow/lite/builtin_ops.h>
#include <tensorflow/lite/c/builtin_op_data.h>
#include <tensorflow/lite/c/common.h>
#include <tensorflow/lite/minimal_logging.h>
#include "tensorflow/lite/delegates/utils.h"

namespace armnnDelegate
{

TfLiteStatus ValidateAddOperator(DelegateData& delegateData,
                                 TfLiteContext* tfLiteContext,
                                 const armnn::TensorInfo& inputInfo1,
                                 const armnn::TensorInfo& inputInfo2,
                                 const armnn::TensorInfo& outputInfo)
{
    bool isSupported = false;
    auto validateFunc = [&](const armnn::TensorInfo& outputTensorInfo, bool& isSupported)
    {
        FORWARD_LAYER_SUPPORT_FUNC(__func__,
                                   tfLiteContext,
                                   IsAdditionSupported,
                                   delegateData.m_Backends,
                                   isSupported,
                                   inputInfo1,
                                   inputInfo2,
                                   outputTensorInfo);
    };

    validateFunc(outputInfo, isSupported);
    return isSupported ? kTfLiteOk : kTfLiteError;
}


TfLiteStatus ValidateDivOperator(DelegateData& delegateData,
                                 TfLiteContext* tfLiteContext,
                                 const armnn::TensorInfo& inputInfo1,
                                 const armnn::TensorInfo& inputInfo2,
                                 const armnn::TensorInfo& outputInfo)
{
    bool isSupported = false;
    auto validateFunc = [&](const armnn::TensorInfo& outputTensorInfo, bool& isSupported)
    {
        FORWARD_LAYER_SUPPORT_FUNC(__func__,
                                   tfLiteContext,
                                   IsDivisionSupported,
                                   delegateData.m_Backends,
                                   isSupported,
                                   inputInfo1,
                                   inputInfo2,
                                   outputTensorInfo);
    };

    validateFunc(outputInfo, isSupported);
    return isSupported ? kTfLiteOk : kTfLiteError;
}

TfLiteStatus ValidateFloorDivOperator(DelegateData& delegateData,
                                      TfLiteContext* tfLiteContext,
                                      const armnn::TensorInfo& inputInfo1,
                                      const armnn::TensorInfo& inputInfo2,
                                      const armnn::TensorInfo& outputInfo)
{
    // need first to validate that the div operator is supported
    // then that the floor operator is supported
    TfLiteStatus status = ValidateDivOperator(delegateData, tfLiteContext, inputInfo1, inputInfo2, outputInfo);
    if (status != kTfLiteOk)
    {
        return status;
    }
    // if the inputs and output of the div are all Signed32 we don't need to add the floor operator afterward.
    if (AreAllSigned32(inputInfo1, inputInfo2, outputInfo))
    {
        return status;
    }
    // in case broadcasting is being done from one of the inputs to the div
    // choose the full sized input tensor to pass to the floor validation routine
    armnn::TensorInfo floorInputInfo = inputInfo1;
    if (inputInfo1.GetNumDimensions() < inputInfo2.GetNumDimensions())
    {
        floorInputInfo = inputInfo2;
    }
    status = ValidateFloorOperator(delegateData, tfLiteContext, floorInputInfo, outputInfo);
    return status;
}

TfLiteStatus ValidateMaximumOperator(DelegateData& delegateData,
                                     TfLiteContext* tfLiteContext,
                                     const armnn::TensorInfo& inputInfo1,
                                     const armnn::TensorInfo& inputInfo2,
                                     const armnn::TensorInfo& outputInfo)
{
    bool isSupported = false;
    auto validateFunc = [&](const armnn::TensorInfo& outputTensorInfo, bool& isSupported)
    {
        FORWARD_LAYER_SUPPORT_FUNC(__func__,
                                   tfLiteContext,
                                   IsMaximumSupported,
                                   delegateData.m_Backends,
                                   isSupported,
                                   inputInfo1,
                                   inputInfo2,
                                   outputTensorInfo);
    };

    validateFunc(outputInfo, isSupported);
    return isSupported ? kTfLiteOk : kTfLiteError;
}

TfLiteStatus ValidateMinimumOperator(DelegateData& delegateData,
                                     TfLiteContext* tfLiteContext,
                                     const armnn::TensorInfo& inputInfo1,
                                     const armnn::TensorInfo& inputInfo2,
                                     const armnn::TensorInfo& outputInfo)
{
    bool isSupported = false;
    auto validateFunc = [&](const armnn::TensorInfo& outputTensorInfo, bool& isSupported)
    {
        FORWARD_LAYER_SUPPORT_FUNC(__func__,
                                   tfLiteContext,
                                   IsMinimumSupported,
                                   delegateData.m_Backends,
                                   isSupported,
                                   inputInfo1,
                                   inputInfo2,
                                   outputTensorInfo);
    };

    validateFunc(outputInfo, isSupported);
    return isSupported ? kTfLiteOk : kTfLiteError;
}

TfLiteStatus ValidateMulOperator(DelegateData& delegateData,
                                 TfLiteContext* tfLiteContext,
                                 const armnn::TensorInfo& inputInfo1,
                                 const armnn::TensorInfo& inputInfo2,
                                 const armnn::TensorInfo& outputInfo)
{
    bool isSupported = false;
    auto validateFunc = [&](const armnn::TensorInfo& outputTensorInfo, bool& isSupported)
    {
        FORWARD_LAYER_SUPPORT_FUNC(__func__,
                                   tfLiteContext,
                                   IsMultiplicationSupported,
                                   delegateData.m_Backends,
                                   isSupported,
                                   inputInfo1,
                                   inputInfo2,
                                   outputTensorInfo);
    };

    validateFunc(outputInfo, isSupported);
    return isSupported ? kTfLiteOk : kTfLiteError;
}

TfLiteStatus ValidateSubOperator(DelegateData& delegateData,
                                 TfLiteContext* tfLiteContext,
                                 const armnn::TensorInfo& inputInfo1,
                                 const armnn::TensorInfo& inputInfo2,
                                 const armnn::TensorInfo& outputInfo)
{
    bool isSupported = false;
    auto validateFunc = [&](const armnn::TensorInfo& outputTensorInfo, bool& isSupported)
    {
        FORWARD_LAYER_SUPPORT_FUNC(__func__,
                                   tfLiteContext,
                                   IsSubtractionSupported,
                                   delegateData.m_Backends,
                                   isSupported,
                                   inputInfo1,
                                   inputInfo2,
                                   outputTensorInfo);
    };

    validateFunc(outputInfo, isSupported);
    return isSupported ? kTfLiteOk : kTfLiteError;
}

std::pair<armnn::IConnectableLayer*, armnn::IConnectableLayer*> AddFloorDivLayer(
    DelegateData& delegateData,
    const armnn::TensorInfo& outputTensorInfo)
{
    armnn::IConnectableLayer* divisionLayer = delegateData.m_Network->AddDivisionLayer();
    // if the output of the div is Signed32 the Floor layer is not required
    if (armnn::DataType::Signed32 == outputTensorInfo.GetDataType())
    {
        return std::make_pair(divisionLayer, divisionLayer);
    }
    armnn::IOutputSlot& outputSlot = divisionLayer->GetOutputSlot(0);
    outputSlot.SetTensorInfo(outputTensorInfo);
    armnn::IConnectableLayer* floorLayer = delegateData.m_Network->AddFloorLayer();
    outputSlot.Connect(floorLayer->GetInputSlot(0));
    return std::make_pair(divisionLayer, floorLayer);
}

TfLiteStatus VisitElementwiseBinaryOperator(DelegateData& delegateData,
                                            TfLiteContext* tfLiteContext,
                                            TfLiteNode* tfLiteNode,
                                            int nodeIndex,
                                            int32_t elementwiseBinaryOperatorCode)
{
    TF_LITE_ENSURE_STATUS(ValidateNumInputs(tfLiteContext, tfLiteNode, 2, nodeIndex));
    TF_LITE_ENSURE_STATUS(ValidateNumOutputs(tfLiteContext, tfLiteNode, 1, nodeIndex));

    const TfLiteTensor* tfLiteTensors = tfLiteContext->tensors;
    const TfLiteTensor& tfLiteInputTensor0 = tfLiteTensors[tfLiteNode->inputs->data[0]];
    if (IsDynamicTensor(tfLiteInputTensor0))
    {
        TF_LITE_MAYBE_KERNEL_LOG(
            tfLiteContext,
            "TfLiteArmnnDelegate: Dynamic input tensors are not supported in operator #%d node #%d: ",
            elementwiseBinaryOperatorCode, nodeIndex);
        return kTfLiteError;
    }

    const TfLiteTensor& tfLiteInputTensor1 = tfLiteTensors[tfLiteNode->inputs->data[1]];
    if (IsDynamicTensor(tfLiteInputTensor1))
    {
        TF_LITE_MAYBE_KERNEL_LOG(
            tfLiteContext,
            "TfLiteArmnnDelegate: Dynamic input tensors are not supported in operator #%d node #%d: ",
            elementwiseBinaryOperatorCode, nodeIndex);
        return kTfLiteError;
    }

    const TfLiteTensor& tfLiteOutputTensor = tfLiteTensors[tfLiteNode->outputs->data[0]];
    if (IsDynamicTensor(tfLiteOutputTensor))
    {
        TF_LITE_MAYBE_KERNEL_LOG(
            tfLiteContext,
            "TfLiteArmnnDelegate: Dynamic output tensors are not supported in operator #%d node #%d: ",
            elementwiseBinaryOperatorCode, nodeIndex);
        return kTfLiteError;
    }

    armnn::TensorInfo inputTensorInfo0 = GetTensorInfoForTfLiteTensor(tfLiteInputTensor0);
    armnn::TensorInfo inputTensorInfo1 = GetTensorInfoForTfLiteTensor(tfLiteInputTensor1);

    const armnn::TensorInfo& outputTensorInfo = GetTensorInfoForTfLiteTensor(tfLiteOutputTensor);

    if (!delegateData.m_Network)
    {
        switch(elementwiseBinaryOperatorCode)
        {
            case kTfLiteBuiltinAdd:
                return ValidateAddOperator(delegateData,
                                           tfLiteContext,
                                           inputTensorInfo0,
                                           inputTensorInfo1,
                                           outputTensorInfo);
            case kTfLiteBuiltinDiv:
                return ValidateDivOperator(delegateData,
                                           tfLiteContext,
                                           inputTensorInfo0,
                                           inputTensorInfo1,
                                           outputTensorInfo);
            case kTfLiteBuiltinFloorDiv:
                return ValidateFloorDivOperator(delegateData,
                                                tfLiteContext,
                                                inputTensorInfo0,
                                                inputTensorInfo1,
                                                outputTensorInfo);
            case kTfLiteBuiltinMaximum:
                return ValidateMaximumOperator(delegateData,
                                               tfLiteContext,
                                               inputTensorInfo0,
                                               inputTensorInfo1,
                                               outputTensorInfo);
            case kTfLiteBuiltinMinimum:
                return ValidateMinimumOperator(delegateData,
                                               tfLiteContext,
                                               inputTensorInfo0,
                                               inputTensorInfo1,
                                               outputTensorInfo);
            case kTfLiteBuiltinMul:
                return ValidateMulOperator(delegateData,
                                           tfLiteContext,
                                           inputTensorInfo0,
                                           inputTensorInfo1,
                                           outputTensorInfo);
            case kTfLiteBuiltinSub:
                return ValidateSubOperator(delegateData,
                                           tfLiteContext,
                                           inputTensorInfo0,
                                           inputTensorInfo1,
                                           outputTensorInfo);
            default:
                return kTfLiteError;
        }
    }

    armnn::IConnectableLayer* elementwiseBinaryLayer = nullptr;
    MultiLayerFacade multiLayer;
    switch(elementwiseBinaryOperatorCode)
    {
        case kTfLiteBuiltinAdd:
            elementwiseBinaryLayer = delegateData.m_Network->AddAdditionLayer();
            break;
        case kTfLiteBuiltinDiv:
            elementwiseBinaryLayer = delegateData.m_Network->AddDivisionLayer();
            break;
        case kTfLiteBuiltinFloorDiv:
            {
                auto layers = AddFloorDivLayer(delegateData, outputTensorInfo);
                multiLayer.AssignValues(layers.first, layers.second);
                elementwiseBinaryLayer = &multiLayer;
            }
            break;
        case kTfLiteBuiltinMaximum:
            elementwiseBinaryLayer = delegateData.m_Network->AddMaximumLayer();
            break;
        case kTfLiteBuiltinMinimum:
            elementwiseBinaryLayer = delegateData.m_Network->AddMinimumLayer();
            break;
        case kTfLiteBuiltinMul:
            elementwiseBinaryLayer = delegateData.m_Network->AddMultiplicationLayer();
            break;
        case kTfLiteBuiltinSub:
            elementwiseBinaryLayer = delegateData.m_Network->AddSubtractionLayer();
            break;
        default:
            return kTfLiteError;
    }
    ARMNN_ASSERT(elementwiseBinaryLayer != nullptr);
    armnn::IOutputSlot& outputSlot = elementwiseBinaryLayer->GetOutputSlot(0);
    outputSlot.SetTensorInfo(outputTensorInfo);

    auto inputsTensorsProcess = ProcessInputs(elementwiseBinaryLayer,
                                              delegateData,
                                              tfLiteContext,
                                              tfLiteNode);
    if (inputsTensorsProcess == kTfLiteError)
    {
        return inputsTensorsProcess;
    }

    auto reshapeLayer = BroadcastTensor(inputTensorInfo0,
                                        inputTensorInfo1,
                                        elementwiseBinaryLayer,
                                        tfLiteContext,
                                        tfLiteNode,
                                        delegateData);
    if (!reshapeLayer)
    {
        return kTfLiteError;
    }

    auto* tfLiteNodeParameters = reinterpret_cast<TfLiteAddParams*>(tfLiteNode->builtin_data);
    if (!tfLiteNodeParameters)
    {
        // No Activation
        return kTfLiteOk;
    }
    // Check activation
    TfLiteFusedActivation activationType = tfLiteNodeParameters->activation;
    return FusedActivation(tfLiteContext, tfLiteNode, activationType, elementwiseBinaryLayer, 0, delegateData);
}

} // namespace armnnDelegate
