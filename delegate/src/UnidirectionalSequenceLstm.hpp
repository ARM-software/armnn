//
// Copyright © 2021 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include "DelegateUtils.hpp"

#include <armnn/LstmParams.hpp>
#include <armnn/Tensor.hpp>
#include <armnn/utility/IgnoreUnused.hpp>

#include <tensorflow/lite/builtin_ops.h>
#include <tensorflow/lite/c/builtin_op_data.h>
#include <tensorflow/lite/c/common.h>
#include <tensorflow/lite/minimal_logging.h>

namespace armnnDelegate
{

TfLiteStatus VisitUnidirectionalSequenceLstmOperator(DelegateData& delegateData,
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

    const auto nodeParams = reinterpret_cast<TfLiteUnidirectionalSequenceLSTMParams *>(tfLiteNode->builtin_data);
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

    // Set the params structure for the AddUnidirectionalSequenceLstmLayer call
    // Please refer to each operand at
    // https://www.tensorflow.org/mlir/tfl_ops#tflunidirectional_sequence_lstm_tflunidirectionalsequencelstmop
    armnn::LstmInputParams params;

    if (!IsOptionalOperandPresent(tfLiteNode, 1))
    {
        params.m_InputToInputWeights = GetConstTensorForTfLiteTensor(tfLiteTensors, tfLiteNode, 1);
    }

    params.m_InputToForgetWeights = GetConstTensorForTfLiteTensor(tfLiteTensors, tfLiteNode, 2);
    params.m_InputToCellWeights = GetConstTensorForTfLiteTensor(tfLiteTensors, tfLiteNode, 3);
    params.m_InputToOutputWeights = GetConstTensorForTfLiteTensor(tfLiteTensors, tfLiteNode, 4);

    // Recurrent weight tensors of size {n_cell, n_output}
    if (!IsOptionalOperandPresent(tfLiteNode, 5))
    {
        params.m_RecurrentToInputWeights = GetConstTensorForTfLiteTensor(tfLiteTensors, tfLiteNode, 5);
    }

    params.m_RecurrentToForgetWeights = GetConstTensorForTfLiteTensor(tfLiteTensors, tfLiteNode, 6);
    params.m_RecurrentToCellWeights = GetConstTensorForTfLiteTensor(tfLiteTensors, tfLiteNode, 7);
    params.m_RecurrentToOutputWeights = GetConstTensorForTfLiteTensor(tfLiteTensors, tfLiteNode, 8);

    // Peephole weights tensors of size {n_cell}, representing a diagonal matrix.
    if (!IsOptionalOperandPresent(tfLiteNode, 9))
    {
        params.m_CellToInputWeights = GetConstTensorForTfLiteTensor(tfLiteTensors, tfLiteNode, 9);
    }

    if (!IsOptionalOperandPresent(tfLiteNode, 10))
    {
        params.m_CellToForgetWeights = GetConstTensorForTfLiteTensor(tfLiteTensors, tfLiteNode, 10);
    }

    if (!IsOptionalOperandPresent(tfLiteNode, 11))
    {
        params.m_CellToOutputWeights = GetConstTensorForTfLiteTensor(tfLiteTensors, tfLiteNode, 11);
    }

    // Gates bias tensors of size {n_cell}
    if (!IsOptionalOperandPresent(tfLiteNode, 12))
    {
        params.m_InputGateBias = GetConstTensorForTfLiteTensor(tfLiteTensors, tfLiteNode, 12);
    }

    params.m_ForgetGateBias = GetConstTensorForTfLiteTensor(tfLiteTensors, tfLiteNode, 13);
    params.m_CellBias = GetConstTensorForTfLiteTensor(tfLiteTensors, tfLiteNode, 14);
    params.m_OutputGateBias = GetConstTensorForTfLiteTensor(tfLiteTensors, tfLiteNode, 15);

    // Projection weight tensor of size {n_output, n_cell}
    if (!IsOptionalOperandPresent(tfLiteNode, 16))
    {
        params.m_ProjectionWeights = GetConstTensorForTfLiteTensor(tfLiteTensors, tfLiteNode, 16);
    }
    // Projection bias tensor of size {n_output}
    if (!IsOptionalOperandPresent(tfLiteNode, 17))
    {
        params.m_ProjectionBias = GetConstTensorForTfLiteTensor(tfLiteTensors, tfLiteNode, 17);
    }

    // These state tensors are defined as variable tensors, and will be modified by this op.
    armnn::TensorInfo outputStateInInfo = GetTensorInfoForTfLiteTensor(tfLiteTensors[tfLiteNode->inputs->data[18]]);
    armnn::TensorInfo cellStateInInfo = GetTensorInfoForTfLiteTensor(tfLiteTensors[tfLiteNode->inputs->data[19]]);

    // Layer norm coefficient tensors of size {n_cell}, representing a diagonal matrix.
    if (tfLiteNode->inputs->size >= 21 && !IsOptionalOperandPresent(tfLiteNode, 20))
    {
        params.m_InputLayerNormWeights = GetConstTensorForTfLiteTensor(tfLiteTensors, tfLiteNode, 20);
    }

    if (tfLiteNode->inputs->size >= 22 && !IsOptionalOperandPresent(tfLiteNode, 21))
    {
        params.m_ForgetLayerNormWeights = GetConstTensorForTfLiteTensor(tfLiteTensors, tfLiteNode, 21);
    }

    if (tfLiteNode->inputs->size >= 23 && !IsOptionalOperandPresent(tfLiteNode, 22))
    {
        params.m_CellLayerNormWeights = GetConstTensorForTfLiteTensor(tfLiteTensors, tfLiteNode, 22);
    }

    if (tfLiteNode->inputs->size >= 24 && !IsOptionalOperandPresent(tfLiteNode, 23))
    {
        params.m_OutputLayerNormWeights = GetConstTensorForTfLiteTensor(tfLiteTensors, tfLiteNode, 23);
    }

    // set the layer descriptor
    armnn::UnidirectionalSequenceLstmDescriptor desc;
    desc.m_ActivationFunc    = NonNegative(nodeParams->activation, nodeIndex);
    desc.m_ClippingThresCell = nodeParams->cell_clip;
    desc.m_ClippingThresProj = nodeParams->proj_clip;
    desc.m_CifgEnabled       = (params.m_InputToInputWeights == nullptr
                                || params.m_RecurrentToInputWeights == nullptr
                                || params.m_InputGateBias == nullptr);
    desc.m_PeepholeEnabled   = (params.m_CellToForgetWeights != nullptr || params.m_CellToOutputWeights != nullptr);
    desc.m_ProjectionEnabled = (params.m_ProjectionWeights != nullptr);
    desc.m_LayerNormEnabled  = (params.m_InputLayerNormWeights != nullptr
                                || params.m_ForgetLayerNormWeights != nullptr
                                || params.m_CellLayerNormWeights != nullptr
                                || params.m_OutputLayerNormWeights != nullptr);
    desc.m_TimeMajor = nodeParams->time_major;

    const armnn::TensorInfo& inputTensorInfo = GetTensorInfoForTfLiteTensor(tfLiteInputTensor);
    const armnn::TensorInfo& outputTensorInfo = GetTensorInfoForTfLiteTensor(tfLiteOutputTensor);

    unsigned int batchSize  = inputTensorInfo.GetShape()[0];
    unsigned int outputSize = outputTensorInfo.GetShape()[2];
    unsigned int numUnits   = cellStateInInfo.GetShape()[1];

    armnn::DataType dataType = inputTensorInfo.GetDataType();
    float qScale = inputTensorInfo.GetQuantizationScale();
    float qOffset = inputTensorInfo.GetQuantizationOffset();

    armnn::TensorInfo scratchBufferTensorInfo({batchSize, numUnits * 3}, dataType, qScale, qOffset);
    if (!desc.m_CifgEnabled)
    {
        scratchBufferTensorInfo = armnn::TensorInfo({batchSize, numUnits * 4}, dataType, qScale, qOffset);
    }
    armnn::TensorInfo cellStateOutTensorInfo({batchSize, numUnits}, dataType, qScale, qOffset);
    armnn::TensorInfo outputStateOutTensorInfo({batchSize, outputSize}, dataType, qScale, qOffset);

    armnn::LstmInputParamsInfo paramsInfo;
    paramsInfo.m_InputToForgetWeights     = &(params.m_InputToForgetWeights->GetInfo());
    paramsInfo.m_InputToCellWeights       = &(params.m_InputToCellWeights->GetInfo());
    paramsInfo.m_InputToOutputWeights     = &(params.m_InputToOutputWeights->GetInfo());
    paramsInfo.m_RecurrentToForgetWeights = &(params.m_RecurrentToForgetWeights->GetInfo());
    paramsInfo.m_RecurrentToCellWeights   = &(params.m_RecurrentToCellWeights->GetInfo());
    paramsInfo.m_RecurrentToOutputWeights = &(params.m_RecurrentToOutputWeights->GetInfo());
    paramsInfo.m_ForgetGateBias           = &(params.m_ForgetGateBias->GetInfo());
    paramsInfo.m_CellBias                 = &(params.m_CellBias->GetInfo());
    paramsInfo.m_OutputGateBias           = &(params.m_OutputGateBias->GetInfo());

    if (!desc.m_CifgEnabled)
    {
        paramsInfo.m_InputToInputWeights = &(params.m_InputToInputWeights->GetInfo());
        paramsInfo.m_RecurrentToInputWeights = &(params.m_RecurrentToInputWeights->GetInfo());
        if (params.m_CellToInputWeights != nullptr)
        {
            paramsInfo.m_CellToInputWeights = &(params.m_CellToInputWeights->GetInfo());
        }
        paramsInfo.m_InputGateBias = &(params.m_InputGateBias->GetInfo());
    }

    if (desc.m_ProjectionEnabled)
    {
        paramsInfo.m_ProjectionWeights = &(params.m_ProjectionWeights->GetInfo());
        if (params.m_ProjectionBias != nullptr)
        {
            paramsInfo.m_ProjectionBias = &(params.m_ProjectionBias->GetInfo());
        }
    }

    if (desc.m_PeepholeEnabled)
    {
        paramsInfo.m_CellToForgetWeights = &(params.m_CellToForgetWeights->GetInfo());
        paramsInfo.m_CellToOutputWeights = &(params.m_CellToOutputWeights->GetInfo());
    }

    if (desc.m_LayerNormEnabled)
    {
        if(!desc.m_CifgEnabled)
        {
            paramsInfo.m_InputLayerNormWeights = &(params.m_InputLayerNormWeights->GetInfo());
        }
        paramsInfo.m_ForgetLayerNormWeights = &(params.m_ForgetLayerNormWeights->GetInfo());
        paramsInfo.m_CellLayerNormWeights = &(params.m_CellLayerNormWeights->GetInfo());
        paramsInfo.m_OutputLayerNormWeights = &(params.m_OutputLayerNormWeights->GetInfo());
    }

    // hiddenStateOutput and cellStateOutput do not present in TfLite UnidirectionalSequenceLstm
    armnn::Optional<armnn::TensorInfo> optionalTensor;

    bool isSupported = false;
    auto validateFunc = [&](const armnn::TensorInfo& outputInfo, bool& isSupported)
    {
        FORWARD_LAYER_SUPPORT_FUNC(__func__,
                                   tfLiteContext,
                                   IsUnidirectionalSequenceLstmSupported,
                                   delegateData.m_Backends,
                                   isSupported,
                                   inputTensorInfo,
                                   outputStateInInfo,
                                   cellStateInInfo,
                                   outputInfo,
                                   optionalTensor,
                                   optionalTensor,
                                   desc,
                                   paramsInfo);
    };

    if (!delegateData.m_Network)
    {
        validateFunc(outputTensorInfo, isSupported);
        return isSupported ? kTfLiteOk : kTfLiteError;
    }

    armnn::IConnectableLayer* layer = delegateData.m_Network->AddUnidirectionalSequenceLstmLayer(desc, params);
    ARMNN_ASSERT(layer != nullptr);

    layer->GetOutputSlot(0).SetTensorInfo(outputTensorInfo);

    // Connect the inputs
    // input_layer
    delegateData.m_OutputSlotForNode[tfLiteNode->inputs->data[0]]->Connect(layer->GetInputSlot(0));
    // cellStateIn
    delegateData.m_OutputSlotForNode[tfLiteNode->inputs->data[18]]->Connect(layer->GetInputSlot(1));
    //outputStateIn
    delegateData.m_OutputSlotForNode[tfLiteNode->inputs->data[19]]->Connect(layer->GetInputSlot(2));

    armnn::IOutputSlot& outputSlot = layer->GetOutputSlot(0);
    delegateData.m_OutputSlotForNode[static_cast<unsigned long>(tfLiteNode->outputs->data[0])] = &outputSlot;
    return kTfLiteOk;
}

} // namespace armnnDelegate