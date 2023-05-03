//
// Copyright © 2023 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include <OpaqueDelegateUtils.hpp>

namespace armnnOpaqueDelegate
{

TfLiteStatus VisitUnidirectionalSequenceLstmOperator(DelegateData& delegateData,
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

    // Set the params structure for the AddUnidirectionalSequenceLstmLayer call
    // Please refer to each operand at
    // https://www.tensorflow.org/mlir/tfl_ops#tflunidirectional_sequence_lstm_tflunidirectionalsequencelstmop
    armnn::LstmInputParams params;

    if (IsOptionalOperandPresent(tfLiteNode, 1))
    {
        params.m_InputToInputWeights = GetConstTensorForTfLiteTensor(tfLiteContext, tfLiteNode, 1);
    }

    params.m_InputToForgetWeights = GetConstTensorForTfLiteTensor(tfLiteContext, tfLiteNode, 2);
    params.m_InputToCellWeights = GetConstTensorForTfLiteTensor(tfLiteContext, tfLiteNode, 3);
    params.m_InputToOutputWeights = GetConstTensorForTfLiteTensor(tfLiteContext, tfLiteNode, 4);

    // Recurrent weight tensors of size {n_cell, n_output}
    if (IsOptionalOperandPresent(tfLiteNode, 5))
    {
        params.m_RecurrentToInputWeights = GetConstTensorForTfLiteTensor(tfLiteContext, tfLiteNode, 5);
    }

    params.m_RecurrentToForgetWeights = GetConstTensorForTfLiteTensor(tfLiteContext, tfLiteNode, 6);
    params.m_RecurrentToCellWeights = GetConstTensorForTfLiteTensor(tfLiteContext, tfLiteNode, 7);
    params.m_RecurrentToOutputWeights = GetConstTensorForTfLiteTensor(tfLiteContext, tfLiteNode, 8);

    // Peephole weights tensors of size {n_cell}, representing a diagonal matrix.
    if (IsOptionalOperandPresent(tfLiteNode, 9))
    {
        params.m_CellToInputWeights = GetConstTensorForTfLiteTensor(tfLiteContext, tfLiteNode, 9);
    }

    if (IsOptionalOperandPresent(tfLiteNode, 10))
    {
        params.m_CellToForgetWeights = GetConstTensorForTfLiteTensor(tfLiteContext, tfLiteNode, 10);
    }

    if (IsOptionalOperandPresent(tfLiteNode, 11))
    {
        params.m_CellToOutputWeights = GetConstTensorForTfLiteTensor(tfLiteContext, tfLiteNode, 11);
    }

    // Gates bias tensors of size {n_cell}
    if (IsOptionalOperandPresent(tfLiteNode, 12))
    {
        params.m_InputGateBias = GetConstTensorForTfLiteTensor(tfLiteContext, tfLiteNode, 12);
    }

    params.m_ForgetGateBias = GetConstTensorForTfLiteTensor(tfLiteContext, tfLiteNode, 13);
    params.m_CellBias = GetConstTensorForTfLiteTensor(tfLiteContext, tfLiteNode, 14);
    params.m_OutputGateBias = GetConstTensorForTfLiteTensor(tfLiteContext, tfLiteNode, 15);

    // Projection weight tensor of size {n_output, n_cell}
    if (IsOptionalOperandPresent(tfLiteNode, 16))
    {
        params.m_ProjectionWeights = GetConstTensorForTfLiteTensor(tfLiteContext, tfLiteNode, 16);
    }
    // Projection bias tensor of size {n_output}
    if (IsOptionalOperandPresent(tfLiteNode, 17))
    {
        params.m_ProjectionBias = GetConstTensorForTfLiteTensor(tfLiteContext, tfLiteNode, 17);
    }

    // These state tensors are defined as variable tensors, and will be modified by this op.
    const TfLiteOpaqueTensor* tfLiteOutputStateIn = TfLiteOpaqueContextGetOpaqueTensor(tfLiteContext, inputTensors[18]);
    if (!IsValid(tfLiteContext, tfLiteOutputStateIn, operatorCode, nodeIndex))
    {
        return kTfLiteError;
    }
    const TfLiteOpaqueTensor* cellStateIn = TfLiteOpaqueContextGetOpaqueTensor(tfLiteContext, inputTensors[19]);
    if (!IsValid(tfLiteContext, cellStateIn, operatorCode, nodeIndex))
    {
        return kTfLiteError;
    }
    armnn::TensorInfo outputStateInInfo = GetTensorInfoForTfLiteOpaqueTensor(tfLiteOutputStateIn);
    armnn::TensorInfo cellStateInInfo = GetTensorInfoForTfLiteOpaqueTensor(cellStateIn);

    // Layer norm coefficient tensors of size {n_cell}, representing a diagonal matrix.
    if (IsOptionalOperandPresent(tfLiteNode, 20))
    {
        params.m_InputLayerNormWeights = GetConstTensorForTfLiteTensor(tfLiteContext, tfLiteNode, 20);
    }

    if (IsOptionalOperandPresent(tfLiteNode, 21))
    {
        params.m_ForgetLayerNormWeights = GetConstTensorForTfLiteTensor(tfLiteContext, tfLiteNode, 21);
    }

    if (IsOptionalOperandPresent(tfLiteNode, 22))
    {
        params.m_CellLayerNormWeights = GetConstTensorForTfLiteTensor(tfLiteContext, tfLiteNode, 22);
    }

    if (IsOptionalOperandPresent(tfLiteNode, 23))
    {
        params.m_OutputLayerNormWeights = GetConstTensorForTfLiteTensor(tfLiteContext, tfLiteNode, 23);
    }

    const auto nodeParams =
            reinterpret_cast<TfLiteUnidirectionalSequenceLSTMParams*>(TfLiteOpaqueNodeGetBuiltinData(tfLiteNode));

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

    // Intermediates tensors aren't accessible through the new Opaque Interface yet, so we have to cast it for now.
    // This should be changed to use the accessor functions once added.
    auto* classicTfliteNode = reinterpret_cast<const TfLiteNode*>(tfLiteNode);

    if (classicTfliteNode->intermediates->size > 3 && desc.m_LayerNormEnabled)
    {
        auto inputIntermediateTensorInfo =
                GetTensorInfoForTfLiteOpaqueTensor(
                        TfLiteOpaqueContextGetOpaqueTensor(tfLiteContext, classicTfliteNode->intermediates->data[0]));
        auto forgetIntermediateTensorInfo =
                GetTensorInfoForTfLiteOpaqueTensor(
                        TfLiteOpaqueContextGetOpaqueTensor(tfLiteContext, classicTfliteNode->intermediates->data[1]));
        auto cellIntermediateTensorInfo =
                GetTensorInfoForTfLiteOpaqueTensor(
                        TfLiteOpaqueContextGetOpaqueTensor(tfLiteContext, classicTfliteNode->intermediates->data[2]));
        auto outputIntermediateTensorInfo =
                GetTensorInfoForTfLiteOpaqueTensor(
                        TfLiteOpaqueContextGetOpaqueTensor(tfLiteContext, classicTfliteNode->intermediates->data[3]));

        desc.m_InputIntermediateScale  = inputIntermediateTensorInfo.GetQuantizationScale();
        desc.m_ForgetIntermediateScale = forgetIntermediateTensorInfo.GetQuantizationScale();
        desc.m_CellIntermediateScale   = cellIntermediateTensorInfo.GetQuantizationScale();
        desc.m_OutputIntermediateScale = outputIntermediateTensorInfo.GetQuantizationScale();
    }
    else
    {
        float defaultIntermediate = std::pow(2, -12);
        desc.m_InputIntermediateScale = defaultIntermediate;
        desc.m_ForgetIntermediateScale = defaultIntermediate;
        desc.m_CellIntermediateScale = defaultIntermediate;
        desc.m_OutputIntermediateScale = defaultIntermediate;
    }
    if (classicTfliteNode->intermediates->size > 4)
    {
        auto hiddenTensorInfo =
                GetTensorInfoForTfLiteOpaqueTensor(
                        TfLiteOpaqueContextGetOpaqueTensor(tfLiteContext, classicTfliteNode->intermediates->data[4]));
        desc.m_HiddenStateScale = hiddenTensorInfo.GetQuantizationScale();
        desc.m_HiddenStateZeroPoint = hiddenTensorInfo.GetQuantizationOffset();
    }

    float defaultIntermediate = std::pow(2, -12);
    desc.m_InputIntermediateScale = defaultIntermediate;
    desc.m_ForgetIntermediateScale = defaultIntermediate;
    desc.m_CellIntermediateScale = defaultIntermediate;
    desc.m_OutputIntermediateScale = defaultIntermediate;

    const armnn::TensorInfo& inputTensorInfo = GetTensorInfoForTfLiteOpaqueTensor(tfLiteInputTensor);
    const armnn::TensorInfo& outputTensorInfo = GetTensorInfoForTfLiteOpaqueTensor(tfLiteOutputTensor, true);

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
    armnn::TensorInfo cellStateOutTensorInfo({batchSize, numUnits},
                                             cellStateInInfo.GetDataType(),
                                             cellStateInInfo.GetQuantizationScale(),
                                             cellStateInInfo.GetQuantizationOffset());

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

    bool isSupported = false;
    armnn::BackendId setBackend;
    auto validateFunc = [&](const armnn::TensorInfo& outputInfo, bool& isSupported)
    {
        FORWARD_LAYER_OPAQUE_SUPPORT_FUNC("UNIDIRECTIONAL_SEQUENCE_LSTM",
                                          tfLiteContext,
                                          IsUnidirectionalSequenceLstmSupported,
                                          delegateData.m_Backends,
                                          isSupported,
                                          setBackend,
                                          inputTensorInfo,
                                          outputStateInInfo,
                                          cellStateInInfo,
                                          outputStateOutTensorInfo,
                                          cellStateOutTensorInfo,
                                          outputInfo,
                                          desc,
                                          paramsInfo);
    };

    if (!delegateData.m_Network)
    {
        validateFunc(outputTensorInfo, isSupported);
        return isSupported ? kTfLiteOk : kTfLiteError;
    }

    armnn::IConnectableLayer* layer = delegateData.m_Network->AddUnidirectionalSequenceLstmLayer(desc, params);
    layer->SetBackendId(setBackend);
    ARMNN_ASSERT(layer != nullptr);

    layer->GetOutputSlot(0).SetTensorInfo(outputStateOutTensorInfo);
    layer->GetOutputSlot(1).SetTensorInfo(cellStateOutTensorInfo);
    layer->GetOutputSlot(2).SetTensorInfo(outputTensorInfo);

    // Connect the inputs
    // input_layer
    delegateData.m_OutputSlotForNode[inputTensors[0]]->Connect(layer->GetInputSlot(0));
    // cellStateIn
    delegateData.m_OutputSlotForNode[inputTensors[18]]->Connect(layer->GetInputSlot(1));
    //outputStateIn
    delegateData.m_OutputSlotForNode[inputTensors[19]]->Connect(layer->GetInputSlot(2));

    armnn::IOutputSlot& outputSlot = layer->GetOutputSlot(2);
    delegateData.m_OutputSlotForNode[static_cast<unsigned long>(outputTensors[0])] = &outputSlot;

    return kTfLiteOk;
}

} // namespace armnnOpaqueDelegate