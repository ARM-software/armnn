//
// Copyright © 2017 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//
#include "QuantizedLstmLayer.hpp"

#include "LayerCloneBase.hpp"

#include <armnn/QuantizedLstmParams.hpp>
#include <armnn/TypesUtils.hpp>
#include <backendsCommon/CpuTensorHandle.hpp>
#include <backendsCommon/WorkloadFactory.hpp>

namespace armnn
{

QuantizedLstmLayer::QuantizedLstmLayer(const char* name)
    : Layer(3, 2, LayerType::QuantizedLstm, name)
{
}

std::unique_ptr<IWorkload> QuantizedLstmLayer::CreateWorkload(const IWorkloadFactory& factory) const
{
    QuantizedLstmQueueDescriptor descriptor;

    // QuantizedLstmLayer parameters - there are no optional params
    descriptor.m_InputToInputWeights  = m_QuantizedLstmParameters.m_InputToInputWeights.get();
    descriptor.m_InputToForgetWeights = m_QuantizedLstmParameters.m_InputToForgetWeights.get();
    descriptor.m_InputToCellWeights   = m_QuantizedLstmParameters.m_InputToCellWeights.get();
    descriptor.m_InputToOutputWeights = m_QuantizedLstmParameters.m_InputToOutputWeights.get();

    descriptor.m_RecurrentToInputWeights  = m_QuantizedLstmParameters.m_RecurrentToInputWeights.get();
    descriptor.m_RecurrentToForgetWeights = m_QuantizedLstmParameters.m_RecurrentToForgetWeights.get();
    descriptor.m_RecurrentToCellWeights   = m_QuantizedLstmParameters.m_RecurrentToCellWeights.get();
    descriptor.m_RecurrentToOutputWeights = m_QuantizedLstmParameters.m_RecurrentToOutputWeights.get();

    descriptor.m_InputGateBias  = m_QuantizedLstmParameters.m_InputGateBias.get();
    descriptor.m_ForgetGateBias = m_QuantizedLstmParameters.m_ForgetGateBias.get();
    descriptor.m_CellBias       = m_QuantizedLstmParameters.m_CellBias.get();
    descriptor.m_OutputGateBias = m_QuantizedLstmParameters.m_OutputGateBias.get();

    SetAdditionalInfo(descriptor);

    return factory.CreateQuantizedLstm(descriptor, PrepInfoAndDesc(descriptor));
}

QuantizedLstmLayer* QuantizedLstmLayer::Clone(Graph& graph) const
{
    auto layer = CloneBase<QuantizedLstmLayer>(graph, GetName());

    layer->m_QuantizedLstmParameters.m_InputToInputWeights = m_QuantizedLstmParameters.m_InputToInputWeights ?
            std::make_unique<ScopedCpuTensorHandle>(*m_QuantizedLstmParameters.m_InputToInputWeights) : nullptr;
    layer->m_QuantizedLstmParameters.m_InputToForgetWeights = m_QuantizedLstmParameters.m_InputToForgetWeights ?
            std::make_unique<ScopedCpuTensorHandle>(*m_QuantizedLstmParameters.m_InputToForgetWeights) : nullptr;
    layer->m_QuantizedLstmParameters.m_InputToCellWeights = m_QuantizedLstmParameters.m_InputToCellWeights ?
            std::make_unique<ScopedCpuTensorHandle>(*m_QuantizedLstmParameters.m_InputToCellWeights) : nullptr;
    layer->m_QuantizedLstmParameters.m_InputToOutputWeights = m_QuantizedLstmParameters.m_InputToOutputWeights ?
            std::make_unique<ScopedCpuTensorHandle>(*m_QuantizedLstmParameters.m_InputToOutputWeights) : nullptr;

    layer->m_QuantizedLstmParameters.m_RecurrentToInputWeights = m_QuantizedLstmParameters.m_RecurrentToInputWeights ?
            std::make_unique<ScopedCpuTensorHandle>(*m_QuantizedLstmParameters.m_RecurrentToInputWeights) : nullptr;
    layer->m_QuantizedLstmParameters.m_RecurrentToForgetWeights = m_QuantizedLstmParameters.m_RecurrentToForgetWeights
            ? std::make_unique<ScopedCpuTensorHandle>(*m_QuantizedLstmParameters.m_RecurrentToForgetWeights) : nullptr;
    layer->m_QuantizedLstmParameters.m_RecurrentToCellWeights = m_QuantizedLstmParameters.m_RecurrentToCellWeights ?
            std::make_unique<ScopedCpuTensorHandle>(*m_QuantizedLstmParameters.m_RecurrentToCellWeights) : nullptr;
    layer->m_QuantizedLstmParameters.m_RecurrentToOutputWeights = m_QuantizedLstmParameters.m_RecurrentToOutputWeights
            ? std::make_unique<ScopedCpuTensorHandle>(*m_QuantizedLstmParameters.m_RecurrentToOutputWeights) : nullptr;

    layer->m_QuantizedLstmParameters.m_InputGateBias = m_QuantizedLstmParameters.m_InputGateBias ?
            std::make_unique<ScopedCpuTensorHandle>(*m_QuantizedLstmParameters.m_InputGateBias) : nullptr;
    layer->m_QuantizedLstmParameters.m_ForgetGateBias = m_QuantizedLstmParameters.m_ForgetGateBias ?
            std::make_unique<ScopedCpuTensorHandle>(*m_QuantizedLstmParameters.m_ForgetGateBias) : nullptr;
    layer->m_QuantizedLstmParameters.m_CellBias = m_QuantizedLstmParameters.m_CellBias ?
            std::make_unique<ScopedCpuTensorHandle>(*m_QuantizedLstmParameters.m_CellBias) : nullptr;
    layer->m_QuantizedLstmParameters.m_OutputGateBias = m_QuantizedLstmParameters.m_OutputGateBias ?
            std::make_unique<ScopedCpuTensorHandle>(*m_QuantizedLstmParameters.m_OutputGateBias) : nullptr;

    return std::move(layer);
}

std::vector<TensorShape> QuantizedLstmLayer::InferOutputShapes(const std::vector<TensorShape>& inputShapes) const
{
    ARMNN_ASSERT(inputShapes.size() == 3);

    // Get input values for validation
    unsigned int numBatches = inputShapes[0][0];
    unsigned int outputSize = inputShapes[1][1];

    std::vector<TensorShape> outShapes;
    outShapes.push_back(TensorShape({numBatches, outputSize})); // cellStateOut
    outShapes.push_back(TensorShape({numBatches, outputSize})); // output

    return outShapes;
}

void QuantizedLstmLayer::ValidateTensorShapesFromInputs()
{
    VerifyLayerConnections(3, CHECK_LOCATION());

    const TensorShape& outputShape = GetOutputSlot(0).GetTensorInfo().GetShape();

    VerifyShapeInferenceType(outputShape, m_ShapeInferenceMethod);

    auto inferredShapes = InferOutputShapes(
    {
        GetInputSlot(0).GetConnection()->GetTensorInfo().GetShape(), // input
        GetInputSlot(1).GetConnection()->GetTensorInfo().GetShape(), // previousCellStateIn
        GetInputSlot(2).GetConnection()->GetTensorInfo().GetShape()  // previousOutputIn
    });

    ARMNN_ASSERT(inferredShapes.size() == 2);

    // Check weights and bias for nullptr
    ARMNN_ASSERT_MSG(m_QuantizedLstmParameters.m_InputToInputWeights != nullptr,
                     "QuantizedLstmLayer: m_QuantizedLstmParameters.m_InputToInputWeights should not be null.");
    ARMNN_ASSERT_MSG(m_QuantizedLstmParameters.m_InputToForgetWeights != nullptr,
                     "QuantizedLstmLayer: m_QuantizedLstmParameters.m_InputToForgetWeights should not be null.");
    ARMNN_ASSERT_MSG(m_QuantizedLstmParameters.m_InputToCellWeights != nullptr,
                     "QuantizedLstmLayer: m_QuantizedLstmParameters.m_InputToCellWeights should not be null.");
    ARMNN_ASSERT_MSG(m_QuantizedLstmParameters.m_InputToOutputWeights != nullptr,
                     "QuantizedLstmLayer: m_QuantizedLstmParameters.m_InputToOutputWeights should not be null.");

    ARMNN_ASSERT_MSG(m_QuantizedLstmParameters.m_RecurrentToInputWeights != nullptr,
                     "QuantizedLstmLayer: m_QuantizedLstmParameters.m_RecurrentToInputWeights should not be null.");
    ARMNN_ASSERT_MSG(m_QuantizedLstmParameters.m_RecurrentToForgetWeights != nullptr,
                     "QuantizedLstmLayer: m_QuantizedLstmParameters.m_RecurrentToForgetWeights should not be null.");
    ARMNN_ASSERT_MSG(m_QuantizedLstmParameters.m_RecurrentToCellWeights != nullptr,
                     "QuantizedLstmLayer: m_QuantizedLstmParameters.m_RecurrentToCellWeights should not be null.");
    ARMNN_ASSERT_MSG(m_QuantizedLstmParameters.m_RecurrentToOutputWeights != nullptr,
                     "QuantizedLstmLayer: m_QuantizedLstmParameters.m_RecurrentToOutputWeights should not be null.");

    ARMNN_ASSERT_MSG(m_QuantizedLstmParameters.m_InputGateBias != nullptr,
                     "QuantizedLstmLayer: m_QuantizedLstmParameters.m_InputGateBias should not be null.");
    ARMNN_ASSERT_MSG(m_QuantizedLstmParameters.m_ForgetGateBias != nullptr,
                     "QuantizedLstmLayer: m_QuantizedLstmParameters.m_ForgetGateBias should not be null.");
    ARMNN_ASSERT_MSG(m_QuantizedLstmParameters.m_CellBias != nullptr,
                     "QuantizedLstmLayer: m_QuantizedLstmParameters.m_CellBias should not be null.");
    ARMNN_ASSERT_MSG(m_QuantizedLstmParameters.m_OutputGateBias != nullptr,
                     "QuantizedLstmLayer: m_QuantizedLstmParameters.m_OutputGateBias should not be null.");

    // Check output TensorShape(s) match inferred shape
    ValidateAndCopyShape(outputShape, inferredShapes[0], m_ShapeInferenceMethod, "QuantizedLstmLayer");

    ValidateAndCopyShape(GetOutputSlot(1).GetTensorInfo().GetShape(),
                         inferredShapes[1],
                         m_ShapeInferenceMethod,
                         "QuantizedLstmLayer",
                         1);
}

Layer::ConstantTensors QuantizedLstmLayer::GetConstantTensorsByRef()
{
    return
    {
        m_QuantizedLstmParameters.m_InputToInputWeights,
        m_QuantizedLstmParameters.m_InputToForgetWeights,
        m_QuantizedLstmParameters.m_InputToCellWeights,
        m_QuantizedLstmParameters.m_InputToOutputWeights,

        m_QuantizedLstmParameters.m_RecurrentToInputWeights,
        m_QuantizedLstmParameters.m_RecurrentToForgetWeights,
        m_QuantizedLstmParameters.m_RecurrentToCellWeights,
        m_QuantizedLstmParameters.m_RecurrentToOutputWeights,

        m_QuantizedLstmParameters.m_InputGateBias,
        m_QuantizedLstmParameters.m_ForgetGateBias,
        m_QuantizedLstmParameters.m_CellBias,
        m_QuantizedLstmParameters.m_OutputGateBias
    };
}

void QuantizedLstmLayer::Accept(ILayerVisitor& visitor) const
{
    QuantizedLstmInputParams inputParams;

    // InputToX weight tensors
    ConstTensor inputToInputWeightsTensor;
    if (m_QuantizedLstmParameters.m_InputToInputWeights != nullptr)
    {
        ConstTensor inputToInputWeightsTensorCopy(m_QuantizedLstmParameters.m_InputToInputWeights->GetTensorInfo(),
                                                  m_QuantizedLstmParameters.m_InputToInputWeights->Map(true));
        inputToInputWeightsTensor = inputToInputWeightsTensorCopy;
        inputParams.m_InputToInputWeights = &inputToInputWeightsTensor;
    }

    ConstTensor inputToForgetWeightsTensor;
    if (m_QuantizedLstmParameters.m_InputToForgetWeights != nullptr)
    {
        ConstTensor inputToForgetWeightsTensorCopy(m_QuantizedLstmParameters.m_InputToForgetWeights->GetTensorInfo(),
                                                   m_QuantizedLstmParameters.m_InputToForgetWeights->Map(true));
        inputToForgetWeightsTensor = inputToForgetWeightsTensorCopy;
        inputParams.m_InputToForgetWeights = &inputToForgetWeightsTensor;
    }

    ConstTensor inputToCellWeightsTensor;
    if (m_QuantizedLstmParameters.m_InputToCellWeights != nullptr)
    {
        ConstTensor inputToCellWeightsTensorCopy(m_QuantizedLstmParameters.m_InputToCellWeights->GetTensorInfo(),
                                                 m_QuantizedLstmParameters.m_InputToCellWeights->Map(true));
        inputToCellWeightsTensor = inputToCellWeightsTensorCopy;
        inputParams.m_InputToCellWeights = &inputToCellWeightsTensor;
    }

    ConstTensor inputToOutputWeightsTensor;
    if (m_QuantizedLstmParameters.m_InputToOutputWeights != nullptr)
    {
        ConstTensor inputToOutputWeightsTensorCopy(m_QuantizedLstmParameters.m_InputToOutputWeights->GetTensorInfo(),
                                                   m_QuantizedLstmParameters.m_InputToOutputWeights->Map(true));
        inputToOutputWeightsTensor = inputToOutputWeightsTensorCopy;
        inputParams.m_InputToOutputWeights = &inputToOutputWeightsTensor;
    }

    // RecurrentToX weight tensors
    ConstTensor recurrentToInputWeightsTensor;
    if (m_QuantizedLstmParameters.m_RecurrentToInputWeights != nullptr)
    {
        ConstTensor recurrentToInputWeightsTensorCopy(
                m_QuantizedLstmParameters.m_RecurrentToInputWeights->GetTensorInfo(),
                m_QuantizedLstmParameters.m_RecurrentToInputWeights->Map(true));
        recurrentToInputWeightsTensor = recurrentToInputWeightsTensorCopy;
        inputParams.m_RecurrentToInputWeights = &recurrentToInputWeightsTensor;
    }

    ConstTensor recurrentToForgetWeightsTensor;
    if (m_QuantizedLstmParameters.m_RecurrentToForgetWeights != nullptr)
    {
        ConstTensor recurrentToForgetWeightsTensorCopy(
                m_QuantizedLstmParameters.m_RecurrentToForgetWeights->GetTensorInfo(),
                m_QuantizedLstmParameters.m_RecurrentToForgetWeights->Map(true));
        recurrentToForgetWeightsTensor = recurrentToForgetWeightsTensorCopy;
        inputParams.m_RecurrentToForgetWeights = &recurrentToForgetWeightsTensor;
    }

    ConstTensor recurrentToCellWeightsTensor;
    if (m_QuantizedLstmParameters.m_RecurrentToCellWeights != nullptr)
    {
        ConstTensor recurrentToCellWeightsTensorCopy(
                m_QuantizedLstmParameters.m_RecurrentToCellWeights->GetTensorInfo(),
                m_QuantizedLstmParameters.m_RecurrentToCellWeights->Map(true));
        recurrentToCellWeightsTensor = recurrentToCellWeightsTensorCopy;
        inputParams.m_RecurrentToCellWeights = &recurrentToCellWeightsTensor;
    }

    ConstTensor recurrentToOutputWeightsTensor;
    if (m_QuantizedLstmParameters.m_RecurrentToOutputWeights != nullptr)
    {
        ConstTensor recurrentToOutputWeightsTensorCopy(
                m_QuantizedLstmParameters.m_RecurrentToOutputWeights->GetTensorInfo(),
                m_QuantizedLstmParameters.m_RecurrentToOutputWeights->Map(true));
        recurrentToOutputWeightsTensor = recurrentToOutputWeightsTensorCopy;
        inputParams.m_RecurrentToOutputWeights = &recurrentToOutputWeightsTensor;
    }

    // Bias tensors
    ConstTensor inputGateBiasTensor;
    if (m_QuantizedLstmParameters.m_InputGateBias != nullptr)
    {
        ConstTensor inputGateBiasTensorCopy(m_QuantizedLstmParameters.m_InputGateBias->GetTensorInfo(),
                                            m_QuantizedLstmParameters.m_InputGateBias->Map(true));
        inputGateBiasTensor = inputGateBiasTensorCopy;
        inputParams.m_InputGateBias = &inputGateBiasTensor;
    }

    ConstTensor forgetGateBiasTensor;
    if (m_QuantizedLstmParameters.m_ForgetGateBias != nullptr)
    {
        ConstTensor forgetGateBiasTensorCopy(m_QuantizedLstmParameters.m_ForgetGateBias->GetTensorInfo(),
                                             m_QuantizedLstmParameters.m_ForgetGateBias->Map(true));
        forgetGateBiasTensor = forgetGateBiasTensorCopy;
        inputParams.m_ForgetGateBias = &forgetGateBiasTensor;
    }

    ConstTensor cellBiasTensor;
    if (m_QuantizedLstmParameters.m_CellBias != nullptr)
    {
        ConstTensor cellBiasTensorCopy(m_QuantizedLstmParameters.m_CellBias->GetTensorInfo(),
                                       m_QuantizedLstmParameters.m_CellBias->Map(true));
        cellBiasTensor = cellBiasTensorCopy;
        inputParams.m_CellBias = &cellBiasTensor;
    }

    ConstTensor outputGateBiasTensor;
    if (m_QuantizedLstmParameters.m_OutputGateBias != nullptr)
    {
        ConstTensor outputGateBiasCopy(m_QuantizedLstmParameters.m_OutputGateBias->GetTensorInfo(),
                                       m_QuantizedLstmParameters.m_OutputGateBias->Map(true));
        outputGateBiasTensor = outputGateBiasCopy;
        inputParams.m_OutputGateBias = &outputGateBiasTensor;
    }

    visitor.VisitQuantizedLstmLayer(this, inputParams, GetName());
}

} // namespace armnn
