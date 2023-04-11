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
#include <tensorflow/lite/c/c_api_opaque.h>
#include <tensorflow/lite/minimal_logging.h>
#include <tensorflow/lite/kernels/kernel_util.h>

namespace
{

// Macro to call an Is<layer_name>Supported function and log caller name together with reason for lack of support
#define FORWARD_LAYER_OPAQUE_SUPPORT_FUNC(opName, tfLiteContext, func, backends, supported, setBackend, ...) \
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
            TF_LITE_OPAQUE_KERNEL_LOG(tfLiteContext, "%s: backend not registered: %s", \
                                      opName, backendId.Get().c_str()); \
        } \
    } \
    if (!supported) \
    { \
        TF_LITE_OPAQUE_KERNEL_LOG(tfLiteContext, "%s: not supported by any specified backend", opName); \
    } \
} \
catch (const armnn::InvalidArgumentException &e) \
{ \
    throw armnn::InvalidArgumentException(e, "Failed to check layer support", CHECK_LOCATION()); \
}

TfLiteStatus ValidateNumInputs(TfLiteOpaqueContext* tfLiteContext,
                               TfLiteOpaqueNode* tfLiteNode,
                               const unsigned int expectedSize,
                               int nodeIndex)
{
    int numInputs = TfLiteOpaqueNodeNumberOfInputs(tfLiteNode);
    if (static_cast<unsigned int>(numInputs) != expectedSize)
    {
        TF_LITE_OPAQUE_MAYBE_KERNEL_LOG(
                tfLiteContext, "TfLiteArmnnOpaqueDelegate: Unexpected number of inputs (%d != %d) in node #%d",
                numInputs, expectedSize, nodeIndex);
        return kTfLiteError;
    }
    return kTfLiteOk;
}

TfLiteStatus ValidateNumOutputs(TfLiteOpaqueContext* tfLiteContext,
                                TfLiteOpaqueNode* tfLiteNode,
                                const unsigned int expectedSize,
                                int nodeIndex)
{
    auto numOutputs = TfLiteOpaqueNodeNumberOfOutputs(tfLiteNode);
    if (static_cast<unsigned int>(numOutputs) != expectedSize)
    {
        TF_LITE_OPAQUE_MAYBE_KERNEL_LOG(
                tfLiteContext, "TfLiteArmnnOpaqueDelegate: Unexpected number of outputs (%d != %d) in node #%d",
                numOutputs, expectedSize, nodeIndex);
        return kTfLiteError;
    }
    return kTfLiteOk;
}

bool IsConstantTensor(const TfLiteOpaqueTensor* tfLiteTensor)
{
    auto tensorAllocationType = TfLiteOpaqueTensorGetAllocationType(tfLiteTensor);
    if (tensorAllocationType == kTfLiteMmapRo)
    {
        return true;
    }
    return false;
}

bool IsDynamicTensor(const TfLiteOpaqueTensor* tfLiteTensor)
{
    auto tensorAllocationType = TfLiteOpaqueTensorGetAllocationType(tfLiteTensor);
    if (tensorAllocationType == kTfLiteDynamic)
    {
        return true;
    }
    return false;
}

bool IsValid(const TfLiteOpaqueTensor* tfLiteTensor)
{
    return tfLiteTensor == nullptr ? false : true;
}

bool IsValid(TfLiteOpaqueContext* tfLiteContext,
             const TfLiteOpaqueTensor* tfLiteTensor,
             int32_t operatorCode,
             int32_t nodeIndex)
{
    if(!IsValid(tfLiteTensor))
    {
        TF_LITE_OPAQUE_MAYBE_KERNEL_LOG(
                tfLiteContext,
                "TfLiteArmnnDelegate: Invalid TfLite tensor in operator #%d node #%d: ",
                operatorCode, nodeIndex);
        return false;
    }
    if (IsDynamicTensor(tfLiteTensor))
    {
        TF_LITE_OPAQUE_MAYBE_KERNEL_LOG(
                tfLiteContext,
                "TfLiteArmnnDelegate: Dynamic tensors are not supported in operator #%d node #%d: ",
                operatorCode, nodeIndex);
        return false;
    }
    return true;
}

bool IsAffineQuantization(const TfLiteOpaqueTensor& tfLiteTensor)
{
    auto quantizationInfo = TfLiteOpaqueTensorGetQuantization(&tfLiteTensor);
    if (quantizationInfo.type == kTfLiteAffineQuantization)
    {
        return true;
    }
    return false;
}

// Connects the layer to the graph
TfLiteStatus Connect(armnn::IConnectableLayer* layer,
                     TfLiteOpaqueContext* tfLiteContext,
                     TfLiteOpaqueNode* tfLiteNode,
                     armnnOpaqueDelegate::DelegateData& data)
{
    // Get array of input indices, inputIndexArray is set from the TfLiteOpaqueNodeInputs function
    // This function turns inputIndexArray into an int array of indices. These indices point to the index of the
    // tensors for each input slot in the node.
    const int* inputIndexArray;
    int numInputs;
    if(TfLiteOpaqueNodeInputs(tfLiteNode, &inputIndexArray, &numInputs) != kTfLiteOk)
    {
        return kTfLiteError;
    }
    // numInputs is set from TfLiteOpaqueNodeInputs.
    if(numInputs != static_cast<int>(layer->GetNumInputSlots()))
    {
        ARMNN_LOG(error) << "Layer: " << layer->GetName() << ": Expected number of input slots does not match actual "
                                                          "number of input slots.";
        return kTfLiteError;
    }
    // Connect the input slots.
    // For each input slot, get the index of the opaque tensor that was allocated for it.
    for (unsigned int inputIndex = 0; inputIndex < layer->GetNumInputSlots(); ++inputIndex)
    {
        if (data.m_OutputSlotForNode[inputIndexArray[inputIndex]] != nullptr)
        {
            data.m_OutputSlotForNode[inputIndexArray[inputIndex]]->Connect(layer->GetInputSlot(inputIndex));
        }
    }

    // Get array of output indices, outputIndexArray is set from the TfLiteOpaqueNodeOutputs function
    // This function turns outputIndexArray into an int array of indices. These indices point to the tensors for
    // each output slot in the node.
    const int* outputIndexArray;
    int numOutputs;
    if(TfLiteOpaqueNodeOutputs(tfLiteNode, &outputIndexArray, &numOutputs) != kTfLiteOk)
    {
        return kTfLiteError;
    }
    // numOutputs is set from TfLiteOpaqueNodeOutputs.
    if(numOutputs != static_cast<int>(layer->GetNumOutputSlots()))
    {
        ARMNN_LOG(error) << "Layer: " << layer->GetName() << ": Expected number of output slots does not match actual "
                                                             "number of output slots.";
        return kTfLiteError;
    }

    // Prepare output slots
    for (unsigned int outputIndex = 0; outputIndex < layer->GetNumOutputSlots(); ++outputIndex)
    {
        armnn::IOutputSlot& outputSlot = layer->GetOutputSlot(outputIndex);
        data.m_OutputSlotForNode[static_cast<unsigned long>(outputIndexArray[outputIndex])] = &outputSlot;
    }

    return kTfLiteOk;
}

TfLiteStatus FusedActivation(TfLiteOpaqueContext* tfLiteContext,
                             TfLiteOpaqueNode* tfLiteNode,
                             TfLiteFusedActivation activationType,
                             armnn::IConnectableLayer* prevLayer,
                             unsigned int outputSlotIndex,
                             armnnOpaqueDelegate::DelegateData& data)
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
        case kTfLiteActReluN1To1:
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
    FORWARD_LAYER_OPAQUE_SUPPORT_FUNC("ACTIVATION",
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

    // Get array of output indices, outputIndexArray is set from the TfLiteOpaqueNodeOutputs function
    // This function turns outputIndexArray into an int array of indices. These indices point to the tensors for
    // each output slot in the node.
    const int* outputIndexArray;
    int numOutputs;
    TfLiteStatus outputStatus = TfLiteOpaqueNodeOutputs(tfLiteNode, &outputIndexArray, &numOutputs);
    if(outputStatus != kTfLiteOk)
    {
        return kTfLiteError;
    }

    // Connect and prepare output slots
    for (unsigned int outputIndex = 0; outputIndex < activationLayer->GetNumOutputSlots(); ++outputIndex)
    {
        data.m_OutputSlotForNode[static_cast<unsigned long>(
                outputIndexArray[outputIndex])]->Connect(activationLayer->GetInputSlot(0));

        armnn::IOutputSlot& outputSlot = activationLayer->GetOutputSlot(outputIndex);
        data.m_OutputSlotForNode[static_cast<unsigned long>(outputIndexArray[outputIndex])] = &outputSlot;
    }
    return kTfLiteOk;
}

armnn::IConnectableLayer* AddReshapeLayer(TfLiteOpaqueContext* tfLiteContext,
                                          TfLiteOpaqueNode* tfLiteNode,
                                          armnn::IConnectableLayer* prevLayer,
                                          armnn::TensorInfo reshapedOutputTensorInfo,
                                          armnn::TensorInfo outputTensorInfo,
                                          armnnOpaqueDelegate::DelegateData& data)
{
    armnn::ReshapeDescriptor desc;
    desc.m_TargetShape = outputTensorInfo.GetShape();

    bool isSupported = false;
    armnn::BackendId setBackend;
    FORWARD_LAYER_OPAQUE_SUPPORT_FUNC("RESHAPE",
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

    // Gather array of indices and it's length, replaces node->outputs->data[i]
    const int* outputIndices = nullptr;
    int numOutputs = 0;

    TfLiteStatus status = TfLiteOpaqueNodeOutputs(tfLiteNode, &outputIndices, &numOutputs);
    if(status != kTfLiteOk)
    {
        throw armnn::Exception("TfLiteArmnnOpaqueDelegate: Unable to gather output information from node.");
    }

    if (static_cast<unsigned int>(numOutputs) != reshapeLayer->GetNumOutputSlots())
    {
        throw armnn::Exception("TfLiteArmnnOpaqueDelegate: Unexpected number of outputs (" +
                               std::to_string(numOutputs) +
                               "!= " +
                               std::to_string(reshapeLayer->GetNumOutputSlots()) +
                               ") in node.");
    }

    // Connect and prepare output slots
    for (unsigned int outputIndex = 0; outputIndex < reshapeLayer->GetNumOutputSlots(); ++outputIndex)
    {
        data.m_OutputSlotForNode[static_cast<unsigned long>(
                outputIndices[outputIndex])]->Connect(reshapeLayer->GetInputSlot(0));

        armnn::IOutputSlot& outputSlot = reshapeLayer->GetOutputSlot(outputIndex);
        data.m_OutputSlotForNode[static_cast<unsigned long>(outputIndices[outputIndex])] = &outputSlot;
    }
    return reshapeLayer;
}

armnn::DataType GetDataType(const TfLiteOpaqueTensor* tfLiteTensor)
{
    switch (TfLiteOpaqueTensorType(tfLiteTensor))
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
            auto quantizationInfo = TfLiteOpaqueTensorGetQuantization(tfLiteTensor);
            if (quantizationInfo.type == kTfLiteAffineQuantization)
            {
                auto* quantization =
                        reinterpret_cast<TfLiteAffineQuantization*>(quantizationInfo.params);

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
            throw armnn::Exception(
                    &"TfLiteArmnnDelegate: Unsupported data type: " [ TfLiteOpaqueTensorType(tfLiteTensor) ]);
    }
}

armnn::TensorInfo GetTensorInfoForTfLiteOpaqueTensor(const TfLiteOpaqueTensor* tfLiteTensor, bool isOutput = false)
{
    armnn::DataType type = GetDataType(tfLiteTensor);
    armnn::TensorInfo ret;

    auto tensorDimensionSize = TfLiteOpaqueTensorNumDims(tfLiteTensor);
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

            if(IsConstantTensor(tfLiteTensor))
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

        for (int32_t i = 0; i < tensorDimensionSize; ++i)
        {
            int32_t dim = TfLiteOpaqueTensorDim(tfLiteTensor, i);

            if (dim == 0)
            {
                dimensionsSpecificity[i] = false;
            }
            tensorDims[i] = static_cast<unsigned int>(dim);
        }

        armnn::TensorShape tensorShape(static_cast<unsigned int>(tensorDimensionSize),
                                       tensorDims.data(),
                                       dimensionsSpecificity);

        if(IsConstantTensor(tfLiteTensor))
        {
            ret = armnn::TensorInfo(tensorShape, type);
            ret.SetConstant(true);
        }
        else
        {
            ret = armnn::TensorInfo(tensorShape, type);
        }
    }

    auto quantizationInfo = TfLiteOpaqueTensorGetQuantization(tfLiteTensor);
    if (quantizationInfo.type == kTfLiteAffineQuantization)
    {
        // get per-channel quantization parameters
        const auto* affineQuantization =
                reinterpret_cast<TfLiteAffineQuantization*>(quantizationInfo.params);
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
        auto quantizationParameters = TfLiteOpaqueTensorGetQuantizationParams(tfLiteTensor);
        ret.SetQuantizationScale(quantizationParameters.scale);
        ret.SetQuantizationOffset(quantizationParameters.zero_point);
    }

    return ret;
}

armnn::ConstTensor CreateConstTensor(const TfLiteOpaqueTensor* tfLiteTensor,
                                     const armnn::TensorInfo& tensorInfo)
{
    auto allocType = TfLiteOpaqueTensorGetAllocationType(tfLiteTensor);
    if (allocType != kTfLiteMmapRo)
    {
        throw armnn::Exception("TfLiteArmnnDelegate: Not constant allocation type: " + std::to_string(allocType));
    }

    return armnn::ConstTensor(tensorInfo, TfLiteOpaqueTensorData(tfLiteTensor));
}

armnn::ConstTensor* GetConstTensorForTfLiteTensor(const TfLiteOpaqueContext* tfLiteContext,
                                                  TfLiteOpaqueNode* tfLiteNode,
                                                  int index)
{
    const TfLiteOpaqueTensor* tfLiteTensor = TfLiteOpaqueNodeGetInput(tfLiteContext, tfLiteNode, index);
    armnn::TensorInfo tensorInfo = GetTensorInfoForTfLiteOpaqueTensor(tfLiteTensor);

    return new armnn::ConstTensor(tensorInfo, TfLiteOpaqueTensorData(tfLiteTensor));
}

bool IsOptionalOperandPresent(TfLiteOpaqueNode* tfLiteNode, const int operandIndex)
{
    // Get array of input indices, inputIndexArray is set from the TfLiteOpaqueNodeInputs function
    // This function turns inputIndexArray into an int array of indices. These indices point to the index of the
    // tensors for each input slot in the node.
    const int* inputIndexArray;
    int numInputs = 0;

    TfLiteStatus status = TfLiteOpaqueNodeInputs(tfLiteNode, &inputIndexArray, &numInputs);
    if(status != kTfLiteOk)
    {
        throw armnn::Exception("TfLiteArmnnOpaqueDelegate: Unable to gather input information from node.");
    }

    // If the inputs array has fewer than operandIndex entries or if the entry at operandIndex has a value of -1 or
    // less then the input is not present.
    if (numInputs > operandIndex && inputIndexArray[operandIndex] >= 0)
    {
        return true;
    }
    return false;
}

TfLiteStatus ProcessInputs(armnn::IConnectableLayer* layer,
                           armnnOpaqueDelegate::DelegateData& delegateData,
                           TfLiteOpaqueContext* tfLiteContext,
                           TfLiteOpaqueNode* tfLiteNode)
{
    // Get array of input indices, inputIndexArray is set from the TfLiteOpaqueNodeInputs function
    // This function turns inputIndexArray into an int array of indices. These indices point to the index of the
    // tensors for each input slot in the node.
    const int* inputIndexArray;
    int numInputs = 0;

    TfLiteStatus status = TfLiteOpaqueNodeInputs(tfLiteNode, &inputIndexArray, &numInputs);
    if(status != kTfLiteOk)
    {
        throw armnn::Exception("TfLiteArmnnOpaqueDelegate: Unable to gather input information from node.");
    }

    // Process input tensors
    // If input tensor is a Constant tensor create a constant layer and connect it to the network
    for (int32_t inputIndex = 0; inputIndex < static_cast<int32_t>(layer->GetNumInputSlots()); ++inputIndex)
    {
        const TfLiteOpaqueTensor* tfLiteInputTensor = TfLiteOpaqueNodeGetInput(tfLiteContext, tfLiteNode, inputIndex);

        if (IsConstantTensor(tfLiteInputTensor))
        {
            armnn::TensorInfo inputTensorInfo = GetTensorInfoForTfLiteOpaqueTensor(tfLiteInputTensor);

            bool isSupported = false;
            armnn::BackendId setBackend;
            FORWARD_LAYER_OPAQUE_SUPPORT_FUNC("CONSTANT",
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

            auto constantInput = CreateConstTensor(tfLiteInputTensor, inputTensorInfo);

            armnn::IConnectableLayer* constantLayer = delegateData.m_Network->AddConstantLayer(constantInput);
            constantLayer->SetBackendId(setBackend);
            armnn::IOutputSlot& outputSlot = constantLayer->GetOutputSlot(0);
            outputSlot.SetTensorInfo(inputTensorInfo);

            delegateData.m_OutputSlotForNode[inputIndexArray[inputIndex]] = &outputSlot;
        }
    }
    return kTfLiteOk;
}

} // namespace anonymous
