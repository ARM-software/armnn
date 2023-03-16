//
// Copyright © 2023 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include <armnn_delegate.hpp>
#include <DelegateUtils.hpp>

#include <armnn/ArmNN.hpp>
#include <armnn/BackendHelper.hpp>
#include <armnn/utility/Assert.hpp>
#include <armnn/utility/NumericCast.hpp>

#include <armnnUtils/Permute.hpp>
#include <armnnUtils/TensorUtils.hpp>

#include <tensorflow/lite/builtin_ops.h>
#include <tensorflow/lite/c/builtin_op_data.h>
#include <tensorflow/lite/c/common.h>
#include <tensorflow/lite/minimal_logging.h>
#include <tensorflow/lite/kernels/kernel_util.h>

namespace
{

// Macro to call an Is<layer_name>Supported function and log caller name together with reason for lack of support
#define FORWARD_LAYER_SUPPORT_FUNC(opName, tfLiteContext, func, backends, supported, setBackend, ...) \
try \
{ \
    for (auto&& backendId : backends) \
    { \
        auto layerSupportObject = armnn::GetILayerSupportByBackendId(backendId); \
        if (layerSupportObject.IsBackendRegistered()) \
        { \
            std::string reasonIfUnsupported; \
            supported = \
                layerSupportObject.func(__VA_ARGS__, armnn::Optional<std::string&>(reasonIfUnsupported)); \
            if (supported) \
            { \
                setBackend = backendId; \
                break; \
            } \
            else \
            { \
                if (reasonIfUnsupported.size() > 0) \
                { \
                    TFLITE_LOG_PROD(tflite::TFLITE_LOG_WARNING, \
                                    "%s: not supported by armnn: %s", opName, reasonIfUnsupported.c_str()); \
                } \
                else \
                { \
                    TFLITE_LOG_PROD(tflite::TFLITE_LOG_WARNING, \
                                    "%s: not supported by armnn", opName); \
                } \
            } \
        } \
        else \
        { \
            TF_LITE_KERNEL_LOG(tfLiteContext, "%s: backend not registered: %s", opName, backendId.Get().c_str()); \
        } \
    } \
    if (!supported) \
    { \
        TF_LITE_KERNEL_LOG(tfLiteContext, "%s: not supported by any specified backend", opName); \
    } \
} \
catch (const armnn::InvalidArgumentException &e) \
{ \
    throw armnn::InvalidArgumentException(e, "Failed to check layer support", CHECK_LOCATION()); \
}

TfLiteStatus ValidateNumInputs(TfLiteContext* tfLiteContext,
                               TfLiteNode* tfLiteNode,
                               const unsigned int expectedSize,
                               int nodeIndex)
{
    auto numInputs = tfLiteNode->inputs->size;
    if (static_cast<unsigned int >(numInputs) != expectedSize)
    {
        TF_LITE_MAYBE_KERNEL_LOG(
            tfLiteContext, "TfLiteArmnnDelegate: Unexpected number of inputs (%d != %d) in node #%d",
            numInputs, expectedSize, nodeIndex);
        return kTfLiteError;
    }
    return kTfLiteOk;
}

TfLiteStatus ValidateNumOutputs(TfLiteContext* tfLiteContext,
                                TfLiteNode* tfLiteNode,
                                const unsigned int expectedSize,
                                int nodeIndex)
{
    auto numOutputs = tfLiteNode->outputs->size;
    if (static_cast<unsigned int >(numOutputs) != expectedSize)
    {
        TF_LITE_MAYBE_KERNEL_LOG(
            tfLiteContext, "TfLiteArmnnDelegate: Unexpected number of outputs (%d != %d) in node #%d",
            numOutputs, expectedSize, nodeIndex);
        return kTfLiteError;
    }
    return kTfLiteOk;
}

bool IsDynamicTensor(const TfLiteTensor& tfLiteTensor)
{
    auto tensorAllocationType = tfLiteTensor.allocation_type;
    if (tensorAllocationType == kTfLiteDynamic)
    {
        return true;
    }
    return false;
}

bool IsValid(const TfLiteTensor* tfLiteTensor)
{
    return tfLiteTensor == nullptr ? false : true;
}

bool IsValid(TfLiteContext* tfLiteContext, const TfLiteTensor& tfLiteTensor, int32_t operatorCode, int32_t nodeIndex)
{
    if(!IsValid(&tfLiteTensor))
    {
        std::cout << "..Is Not Valid" << std::endl;
        TF_LITE_MAYBE_KERNEL_LOG(
            tfLiteContext,
            "TfLiteArmnnDelegate: Invalid TfLite tensor in operator #%d node #%d: ",
            operatorCode, nodeIndex);
        return false;
    }
    if (IsDynamicTensor(tfLiteTensor))
    {
        std::cout << "..IsDynamicTensor" << std::endl;
        TF_LITE_MAYBE_KERNEL_LOG(
            tfLiteContext,
            "TfLiteArmnnDelegate: Dynamic tensors are not supported in operator #%d node #%d: ",
            operatorCode, nodeIndex);
        return false;
    }
    return true;
}

bool IsAffineQuantization(const TfLiteTensor& tfLiteTensor)
{
    auto quantizationInfo = tfLiteTensor.quantization;
    if (quantizationInfo.type == kTfLiteAffineQuantization)
    {
        return true;
    }
    return false;
}

TfLiteStatus Connect(armnn::IConnectableLayer* layer,
                     TfLiteNode* tfLiteNode,
                     armnnDelegate::DelegateData& data)
{
    ARMNN_ASSERT(static_cast<unsigned int>(tfLiteNode->outputs->size) == layer->GetNumOutputSlots());

    // Connect the input slots
    for (unsigned int inputIndex = 0; inputIndex < layer->GetNumInputSlots(); ++inputIndex)
    {
        if (data.m_OutputSlotForNode[tfLiteNode->inputs->data[inputIndex]] != nullptr)
        {
            data.m_OutputSlotForNode[tfLiteNode->inputs->data[inputIndex]]->Connect(layer->GetInputSlot(inputIndex));
        }
    }

    // Prepare output slots
    for (unsigned int outputIndex = 0; outputIndex < layer->GetNumOutputSlots(); ++outputIndex)
    {
        armnn::IOutputSlot& outputSlot = layer->GetOutputSlot(outputIndex);
        data.m_OutputSlotForNode[static_cast<unsigned long>(tfLiteNode->outputs->data[outputIndex])] = &outputSlot;
    }

    return kTfLiteOk;
}

TfLiteStatus FusedActivation(TfLiteContext* tfLiteContext,
                             TfLiteNode* tfLiteNode,
                             TfLiteFusedActivation activationType,
                             armnn::IConnectableLayer* prevLayer,
                             unsigned int outputSlotIndex,
                             armnnDelegate::DelegateData& data)
{

    const armnn::TensorInfo& activationOutputInfo = prevLayer->GetOutputSlot(outputSlotIndex).GetTensorInfo();

    armnn::ActivationDescriptor activationDesc;

    switch (activationType)
    {
        case kTfLiteActNone:
        {
            // No Activation
            return kTfLiteOk;
        }
        case kTfLiteActRelu:
        {
            activationDesc.m_Function = armnn::ActivationFunction::ReLu;
            break;
        }
// The name of kTfLiteActRelu1 changed after TF Lite v2.3
#if defined(ARMNN_POST_TFLITE_2_3)
        case kTfLiteActReluN1To1:
#else
        case kTfLiteActRelu1:
#endif
        {
            activationDesc.m_Function = armnn::ActivationFunction::BoundedReLu;
            activationDesc.m_A = 1.0f;
            activationDesc.m_B = -1.0f;
            break;
        }
        case kTfLiteActRelu6:
        {
            activationDesc.m_Function = armnn::ActivationFunction::BoundedReLu;
            activationDesc.m_A = 6.0f;
            activationDesc.m_B = 0.0f;
            break;
        }
        case kTfLiteActSigmoid:
        {
            activationDesc.m_Function = armnn::ActivationFunction::Sigmoid;
            break;
        }
        case kTfLiteActTanh:
        {
            activationDesc.m_Function = armnn::ActivationFunction::TanH;
            activationDesc.m_A = 1.0f;
            activationDesc.m_B = 1.0f;
            break;
        }
        default:
            return kTfLiteError;
    }

    bool isSupported = false;
    armnn::BackendId setBackend;
    FORWARD_LAYER_SUPPORT_FUNC("ACTIVATION",
                               tfLiteContext,
                               IsActivationSupported,
                               data.m_Backends,
                               isSupported,
                               setBackend,
                               activationOutputInfo,
                               activationOutputInfo,
                               activationDesc);
    if (!isSupported)
    {
        return kTfLiteError;
    }
    armnn::IConnectableLayer* activationLayer = data.m_Network->AddActivationLayer(activationDesc);
    activationLayer->SetBackendId(setBackend);

    ARMNN_ASSERT(activationLayer != nullptr);
    activationLayer->GetOutputSlot(0).SetTensorInfo(activationOutputInfo);

    // Connect and prepare output slots
    for (unsigned int outputIndex = 0; outputIndex < activationLayer->GetNumOutputSlots(); ++outputIndex)
    {
        data.m_OutputSlotForNode[static_cast<unsigned long>(
                tfLiteNode->outputs->data[outputIndex])]->Connect(activationLayer->GetInputSlot(0));
        armnn::IOutputSlot& outputSlot = activationLayer->GetOutputSlot(outputIndex);
        data.m_OutputSlotForNode[static_cast<unsigned long>(
                tfLiteNode->outputs->data[outputIndex])] = &outputSlot;
    }
    return kTfLiteOk;
}

armnn::IConnectableLayer* AddReshapeLayer(TfLiteContext* tfLiteContext,
                                          TfLiteNode* tfLiteNode,
                                          armnn::IConnectableLayer* prevLayer,
                                          armnn::TensorInfo reshapedOutputTensorInfo,
                                          armnn::TensorInfo outputTensorInfo,
                                          armnnDelegate::DelegateData& data)
{
    armnn::ReshapeDescriptor desc;
    desc.m_TargetShape = outputTensorInfo.GetShape();

    bool isSupported = false;
    armnn::BackendId setBackend;
    FORWARD_LAYER_SUPPORT_FUNC("RESHAPE",
                               tfLiteContext,
                               IsReshapeSupported,
                               data.m_Backends,
                               isSupported,
                               setBackend,
                               reshapedOutputTensorInfo,
                               outputTensorInfo,
                               desc);

    if (!isSupported)
    {
        return nullptr;
    }

    armnn::IConnectableLayer* reshapeLayer = data.m_Network->AddReshapeLayer(desc);
    reshapeLayer->SetBackendId(setBackend);
    ARMNN_ASSERT(reshapeLayer != nullptr);

    prevLayer->GetOutputSlot(0).SetTensorInfo(reshapedOutputTensorInfo);
    reshapeLayer->GetOutputSlot(0).SetTensorInfo(outputTensorInfo);

    // Connect and prepare output slots
    for (unsigned int outputIndex = 0; outputIndex < reshapeLayer->GetNumOutputSlots(); ++outputIndex)
    {
        data.m_OutputSlotForNode[static_cast<unsigned long>(
                tfLiteNode->outputs->data[outputIndex])]->Connect(reshapeLayer->GetInputSlot(0));
        armnn::IOutputSlot& outputSlot = reshapeLayer->GetOutputSlot(outputIndex);
        data.m_OutputSlotForNode[static_cast<unsigned long>(
                tfLiteNode->outputs->data[outputIndex])] = &outputSlot;
    }
    return reshapeLayer;
}

armnn::DataType GetDataType(const TfLiteTensor& tfLiteTensor)
{
    switch (tfLiteTensor.type)
    {
        case kTfLiteBool:
            return armnn::DataType::Boolean;
        case kTfLiteFloat32:
            return armnn::DataType::Float32;
        case kTfLiteFloat16:
            return armnn::DataType::Float16;
        case kTfLiteUInt8:
            return armnn::DataType::QAsymmU8;
        case kTfLiteInt8:
        {
            auto quantizationInfo = tfLiteTensor.quantization;
            if (quantizationInfo.type == kTfLiteAffineQuantization)
            {
                auto* quantization =
                    reinterpret_cast<TfLiteAffineQuantization*>(tfLiteTensor.quantization.params);
                if (quantization->zero_point != nullptr && quantization->zero_point->size == 1)
                {
                    return armnn::DataType::QAsymmS8;
                }
                else
                {
                    return armnn::DataType::QSymmS8;
                }
            }
            else
            {
                return armnn::DataType::QAsymmS8;
            }
        }
        case kTfLiteInt16:
            return armnn::DataType::QSymmS16;
        case kTfLiteInt32:
            return armnn::DataType::Signed32;
        case kTfLiteInt64:
            return armnn::DataType::Signed64;
        default:
            throw armnn::Exception(&"TfLiteArmnnDelegate: Unsupported data type: " [ tfLiteTensor.type]);
    }
}

armnn::TensorInfo GetTensorInfoForTfLiteTensor(const TfLiteTensor& tfLiteTensor, bool isOutput = false)
{
    armnn::DataType type = GetDataType(tfLiteTensor);
    armnn::TensorInfo ret;
    auto tensorDimensionSize = tfLiteTensor.dims->size;
    if (tensorDimensionSize == 0)
    {
        // If input tensor does not have a shape
        // assuming that it has 1D tensor
        if (!isOutput)
        {
            std::vector<unsigned int> safeShape = { 1 };
            bool dimensionsSpecificity[1] = { true };
            armnn::TensorShape tensorShape(armnn::numeric_cast<unsigned int>(safeShape.size()),
                                           safeShape.data(),
                                           dimensionsSpecificity);
            ret = armnn::TensorInfo(tensorShape, type);
            if(tflite::IsConstantTensor(&tfLiteTensor))
            {
                ret.SetConstant(true);
            }
        }
        else
        {
            armnn::TensorShape tensorShape(armnn::Dimensionality::NotSpecified);
            ret = armnn::TensorInfo(tensorShape, type);
        }
    }
    else
    {
        std::vector<unsigned int> tensorDims(static_cast<unsigned int>(tensorDimensionSize));
        bool dimensionsSpecificity[5] = { true, true, true, true, true };
        for (unsigned int i = 0; i < static_cast<unsigned int>(tensorDimensionSize); ++i) {
            auto dim = tfLiteTensor.dims->data[i];
            if (dim == 0)
            {
                dimensionsSpecificity[i] = false;
            }
            tensorDims[i] = static_cast<unsigned int>(dim);
        }
        armnn::TensorShape tensorShape(static_cast<unsigned int>(tensorDimensionSize),
                                       tensorDims.data(),
                                       dimensionsSpecificity);

        if(tflite::IsConstantTensor(&tfLiteTensor))
        {
            ret = armnn::TensorInfo(tensorShape, type);
            ret.SetConstant(true);
        }
        else
        {
            ret = armnn::TensorInfo(tensorShape, type);
        }
    }

    auto quantizationInfo = tfLiteTensor.quantization;
    if (quantizationInfo.type == kTfLiteAffineQuantization)
    {
        // get per-channel quantization parameters
        const auto* affineQuantization =
            reinterpret_cast<TfLiteAffineQuantization*>(tfLiteTensor.quantization.params);
        if (affineQuantization->scale->size > 1)
        {
            std::vector<float> quantizationScales;
            for (unsigned int i = 0; i < static_cast<unsigned int>(affineQuantization->scale->size); ++i)
            {
                quantizationScales.push_back(affineQuantization->scale->data[i]);
            }
            ret.SetQuantizationScales(quantizationScales);
            ret.SetQuantizationDim(armnn::numeric_cast<unsigned int>(affineQuantization->quantized_dimension));
        }
        else
        {
            ret.SetQuantizationScale(affineQuantization->scale->data[0]);
            ret.SetQuantizationOffset(affineQuantization->zero_point->data[0]);
        }
    }
    else
    {
        auto quantizationParameters = tfLiteTensor.params;
        ret.SetQuantizationScale(quantizationParameters.scale);
        ret.SetQuantizationOffset(quantizationParameters.zero_point);
    }

    return ret;
}

armnn::ConstTensor CreateConstTensor(const TfLiteTensor* tfLiteTensor,
                                     const armnn::TensorInfo& tensorInfo)
{
    if (tfLiteTensor->allocation_type != kTfLiteMmapRo)
    {
        throw armnn::Exception(
            "TfLiteArmnnDelegate:  Not constant allocation type: " + std::to_string(tfLiteTensor->allocation_type));
    }

    return armnn::ConstTensor(tensorInfo, tfLiteTensor->data.data);
}

armnn::ConstTensor* GetConstTensorForTfLiteTensor(const TfLiteTensor* tfLiteTensors, TfLiteNode* tfLiteNode, int index)
{
    const TfLiteTensor &tfLiteTensor = tfLiteTensors[tfLiteNode->inputs->data[index]];
    armnn::TensorInfo tensorInfo = GetTensorInfoForTfLiteTensor(tfLiteTensor);
    return new armnn::ConstTensor(tensorInfo, tfLiteTensor.data.data);
}

bool IsOptionalOperandPresent(TfLiteNode* tfLiteNode, const int operandIndex)
{
    // If the inputs array has fewer than operandIndex entries or if the entry at operandIndex has a value of -1 or
    // less then the input is not present.
    if (tfLiteNode->inputs->size > operandIndex && tfLiteNode->inputs->data[operandIndex] >= 0)
    {
        return true;
    }
    return false;
}

TfLiteStatus ProcessInputs(armnn::IConnectableLayer* layer,
                           armnnDelegate::DelegateData& delegateData,
                           TfLiteContext* tfLiteContext,
                           TfLiteNode* tfLiteNode)
{
    const TfLiteTensor* tfLiteTensors = tfLiteContext->tensors;
    // Process input tensors
    // If input tensor is a Constant tensor create a constant layer and connect it to the network
    for (unsigned int inputIndex = 0; inputIndex < layer->GetNumInputSlots(); ++inputIndex)
    {
        const TfLiteTensor& tfLiteInputTensor = tfLiteTensors[tfLiteNode->inputs->data[inputIndex]];
        if (tflite::IsConstantTensor(&tfLiteInputTensor))
        {
            armnn::TensorInfo inputTensorInfo = GetTensorInfoForTfLiteTensor(tfLiteInputTensor);
            bool isSupported = false;
            armnn::BackendId setBackend;
            FORWARD_LAYER_SUPPORT_FUNC("CONSTANT",
                                       tfLiteContext,
                                       IsConstantSupported,
                                       delegateData.m_Backends,
                                       isSupported,
                                       setBackend,
                                       inputTensorInfo);
            if (!isSupported)
            {
                return kTfLiteError;
            }
            auto constantInput = CreateConstTensor(&tfLiteInputTensor,
                                                   inputTensorInfo);
            armnn::IConnectableLayer* constantLayer = delegateData.m_Network->AddConstantLayer(constantInput);
            constantLayer->SetBackendId(setBackend);
            armnn::IOutputSlot& outputSlot = constantLayer->GetOutputSlot(0);
            outputSlot.SetTensorInfo(inputTensorInfo);

            delegateData.m_OutputSlotForNode[tfLiteNode->inputs->data[inputIndex]] = &outputSlot;
        }
    }
    return kTfLiteOk;
}

} // namespace anonymous
