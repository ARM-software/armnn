//
// Copyright © 2022-2023 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include <armnn/utility/IgnoreUnused.hpp>

#include <tensorflow/lite/builtin_ops.h>
#include <tensorflow/lite/c/builtin_op_data.h>
#include <tensorflow/lite/c/common.h>
#include <tensorflow/lite/minimal_logging.h>
#include <tensorflow/lite/kernels/internal/tensor_ctypes.h>

namespace armnnDelegate
{

TfLiteStatus VisitTransposeOperator(DelegateData& delegateData,
                                    TfLiteContext* tfLiteContext,
                                    TfLiteNode* tfLiteNode,
                                    int nodeIndex,
                                    int32_t tfliteTransposeOperatorCode)
{
    TF_LITE_ENSURE_STATUS(ValidateNumInputs(tfLiteContext, tfLiteNode, 2, nodeIndex));
    TF_LITE_ENSURE_STATUS(ValidateNumOutputs(tfLiteContext, tfLiteNode, 1, nodeIndex));

    const TfLiteTensor *tfLiteTensors = tfLiteContext->tensors;
    const TfLiteTensor& tfLiteInputTensor0 = tfLiteTensors[tfLiteNode->inputs->data[0]];
    if (IsDynamicTensor(tfLiteInputTensor0))
    {
        TF_LITE_MAYBE_KERNEL_LOG(tfLiteContext,
                                 "TfLiteArmnnDelegate: Dynamic input tensors are not supported in "
                                 "operator #%d node #%d: ",
                                 tfliteTransposeOperatorCode, nodeIndex);

        return kTfLiteError;
    }

    const TfLiteTensor& tfLiteInputTensor1 = tfLiteTensors[tfLiteNode->inputs->data[1]];
    if (IsDynamicTensor(tfLiteInputTensor1))
    {
        TF_LITE_MAYBE_KERNEL_LOG(tfLiteContext,
                                 "TfLiteArmnnDelegate: Dynamic input tensors are not supported in "
                                 "operator #%d node #%d: ",
                                 tfliteTransposeOperatorCode, nodeIndex);
        return kTfLiteError;
    }

    const TfLiteTensor& tfLiteOutputTensor = tfLiteTensors[tfLiteNode->outputs->data[0]];
    if (IsDynamicTensor(tfLiteOutputTensor))
    {
        TF_LITE_MAYBE_KERNEL_LOG(tfLiteContext,
                                 "TfLiteArmnnDelegate: Dynamic output tensors are not supported in "
                                 "operator #%d node #%d: ",
                                 tfliteTransposeOperatorCode, nodeIndex);
        return kTfLiteError;
    }

    const armnn::TensorInfo& inputTensorInfo0 = GetTensorInfoForTfLiteTensor(tfLiteInputTensor0);
    const armnn::TensorInfo& outputTensorInfo = GetTensorInfoForTfLiteTensor(tfLiteOutputTensor, true);

    auto* permTensorDataPtr = tflite::GetTensorData<int32_t>(&tfLiteInputTensor1);
    unsigned int numEl = tfLiteInputTensor1.dims->data[0];

    if (numEl > static_cast<int>(armnn::MaxNumOfTensorDimensions))
    {
        return kTfLiteError;
    }

    if (tfLiteInputTensor1.dims->size != 1)
    {
        return kTfLiteError;
    }

    armnn::TransposeDescriptor descriptor(armnn::PermutationVector(
        reinterpret_cast<const armnn::PermutationVector::ValueType *> (permTensorDataPtr),
        static_cast<armnn::PermutationVector::SizeType>(numEl)));

    bool isSupported = false;
    armnn::BackendId setBackend;
    auto validateFunc = [&](const armnn::TensorInfo& outputTensorInfo, bool& isSupported)
    {
        FORWARD_LAYER_SUPPORT_FUNC("TRANSPOSE",
                                   tfLiteContext,
                                   IsTransposeSupported,
                                   delegateData.m_Backends,
                                   isSupported,
                                   setBackend,
                                   inputTensorInfo0,
                                   outputTensorInfo,
                                   descriptor);
    };

    if (!delegateData.m_Network)
    {
        validateFunc(outputTensorInfo, isSupported);
        return isSupported ? kTfLiteOk : kTfLiteError;
    }

    auto layerName = GetLayerName(armnn::LayerType::Transpose, nodeIndex);
    armnn::IConnectableLayer* transposeLayer = delegateData.m_Network->AddTransposeLayer(descriptor, layerName.c_str());
    transposeLayer->SetBackendId(setBackend);
    ARMNN_ASSERT(transposeLayer != nullptr);

     // permutation vector given to descriptor object
    if (transposeLayer->GetNumInputSlots() != 1)
    {
        return kTfLiteError;
    }

    armnn::IOutputSlot& outputSlot = transposeLayer->GetOutputSlot(0);
    outputSlot.SetTensorInfo(outputTensorInfo);

    // try to connect the Constant Inputs if there are any
    if (ProcessInputs(transposeLayer, delegateData, tfLiteContext, tfLiteNode, nodeIndex) != kTfLiteOk)
    {
        return kTfLiteError;
    }

    return Connect(transposeLayer, tfLiteNode, delegateData);
}
} // namespace armnnDelegate
