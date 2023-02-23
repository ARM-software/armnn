//
// Copyright © 2017 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//
#include "QuantizedLstmLayer.hpp"

#include "LayerCloneBase.hpp"

#include <armnn/QuantizedLstmParams.hpp>
#include <armnn/TypesUtils.hpp>
#include <armnn/backends/TensorHandle.hpp>
#include <armnn/backends/WorkloadFactory.hpp>

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

    return factory.CreateWorkload(LayerType::QuantizedLstm, descriptor, PrepInfoAndDesc(descriptor));
}

QuantizedLstmLayer* QuantizedLstmLayer::Clone(Graph& graph) const
{
    auto layer = CloneBase<QuantizedLstmLayer>(graph, GetName());

    layer->m_QuantizedLstmParameters.m_InputToInputWeights = m_QuantizedLstmParameters.m_InputToInputWeights ?
            m_QuantizedLstmParameters.m_InputToInputWeights : nullptr;
    layer->m_QuantizedLstmParameters.m_InputToForgetWeights = m_QuantizedLstmParameters.m_InputToForgetWeights ?
            m_QuantizedLstmParameters.m_InputToForgetWeights : nullptr;
    layer->m_QuantizedLstmParameters.m_InputToCellWeights = m_QuantizedLstmParameters.m_InputToCellWeights ?
            m_QuantizedLstmParameters.m_InputToCellWeights : nullptr;
    layer->m_QuantizedLstmParameters.m_InputToOutputWeights = m_QuantizedLstmParameters.m_InputToOutputWeights ?
            m_QuantizedLstmParameters.m_InputToOutputWeights : nullptr;

    layer->m_QuantizedLstmParameters.m_RecurrentToInputWeights = m_QuantizedLstmParameters.m_RecurrentToInputWeights ?
            m_QuantizedLstmParameters.m_RecurrentToInputWeights : nullptr;
    layer->m_QuantizedLstmParameters.m_RecurrentToForgetWeights = m_QuantizedLstmParameters.m_RecurrentToForgetWeights
            ? m_QuantizedLstmParameters.m_RecurrentToForgetWeights : nullptr;
    layer->m_QuantizedLstmParameters.m_RecurrentToCellWeights = m_QuantizedLstmParameters.m_RecurrentToCellWeights ?
            m_QuantizedLstmParameters.m_RecurrentToCellWeights : nullptr;
    layer->m_QuantizedLstmParameters.m_RecurrentToOutputWeights = m_QuantizedLstmParameters.m_RecurrentToOutputWeights
            ? m_QuantizedLstmParameters.m_RecurrentToOutputWeights : nullptr;

    layer->m_QuantizedLstmParameters.m_InputGateBias = m_QuantizedLstmParameters.m_InputGateBias ?
            m_QuantizedLstmParameters.m_InputGateBias : nullptr;
    layer->m_QuantizedLstmParameters.m_ForgetGateBias = m_QuantizedLstmParameters.m_ForgetGateBias ?
            m_QuantizedLstmParameters.m_ForgetGateBias : nullptr;
    layer->m_QuantizedLstmParameters.m_CellBias = m_QuantizedLstmParameters.m_CellBias ?
            m_QuantizedLstmParameters.m_CellBias : nullptr;
    layer->m_QuantizedLstmParameters.m_OutputGateBias = m_QuantizedLstmParameters.m_OutputGateBias ?
            m_QuantizedLstmParameters.m_OutputGateBias : nullptr;

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

Layer::ImmutableConstantTensors QuantizedLstmLayer::GetConstantTensorsByRef() const
{
    // For API stability DO NOT ALTER order and add new members to the end of vector
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

void QuantizedLstmLayer::ExecuteStrategy(IStrategy& strategy) const
{
    std::vector<ConstTensor> constTensors;

    ManagedConstTensorHandle managedInputToInputWeights(m_QuantizedLstmParameters.m_InputToInputWeights);
    ManagedConstTensorHandle managedInputToForgetWeights(m_QuantizedLstmParameters.m_InputToForgetWeights);
    ManagedConstTensorHandle managedInputToCellWeights(m_QuantizedLstmParameters.m_InputToCellWeights);
    ManagedConstTensorHandle managedInputToOutputWeights(m_QuantizedLstmParameters.m_InputToOutputWeights);

    ManagedConstTensorHandle managedRecurrentToInputWeights(m_QuantizedLstmParameters.m_RecurrentToInputWeights);
    ManagedConstTensorHandle managedRecurrentToForgetWeights(m_QuantizedLstmParameters.m_RecurrentToForgetWeights);
    ManagedConstTensorHandle managedRecurrentToCellWeights(m_QuantizedLstmParameters.m_RecurrentToCellWeights);
    ManagedConstTensorHandle managedRecurrentToOutputWeights(m_QuantizedLstmParameters.m_RecurrentToOutputWeights);

    ManagedConstTensorHandle managedInputGateBias(m_QuantizedLstmParameters.m_InputGateBias);
    ManagedConstTensorHandle managedForgetGateBias(m_QuantizedLstmParameters.m_ForgetGateBias);
    ManagedConstTensorHandle managedCellBias(m_QuantizedLstmParameters.m_CellBias);
    ManagedConstTensorHandle managedOutputGateBias(m_QuantizedLstmParameters.m_OutputGateBias);

    // InputToX weight tensors
    if (m_QuantizedLstmParameters.m_InputToInputWeights != nullptr)
    {
        constTensors.emplace_back(ConstTensor(managedInputToInputWeights.GetTensorInfo(),
                                              managedInputToInputWeights.Map()));
    }

    if (m_QuantizedLstmParameters.m_InputToForgetWeights != nullptr)
    {
        constTensors.emplace_back(ConstTensor(managedInputToForgetWeights.GetTensorInfo(),
                                              managedInputToForgetWeights.Map()));
    }

    if (m_QuantizedLstmParameters.m_InputToCellWeights != nullptr)
    {
        constTensors.emplace_back(ConstTensor(managedInputToCellWeights.GetTensorInfo(),
                                              managedInputToCellWeights.Map()));
    }

    if (m_QuantizedLstmParameters.m_InputToOutputWeights != nullptr)
    {
        constTensors.emplace_back(ConstTensor(managedInputToOutputWeights.GetTensorInfo(),
                                              managedInputToOutputWeights.Map()));
    }

    // RecurrentToX weight tensors
    if (m_QuantizedLstmParameters.m_RecurrentToInputWeights != nullptr)
    {
        constTensors.emplace_back(ConstTensor(
                managedRecurrentToInputWeights.GetTensorInfo(),
                managedRecurrentToInputWeights.Map()));
    }

    if (m_QuantizedLstmParameters.m_RecurrentToForgetWeights != nullptr)
    {
        constTensors.emplace_back(ConstTensor(
                managedRecurrentToForgetWeights.GetTensorInfo(),
                managedRecurrentToForgetWeights.Map()));
    }

    if (m_QuantizedLstmParameters.m_RecurrentToCellWeights != nullptr)
    {
        constTensors.emplace_back(ConstTensor(
                managedRecurrentToCellWeights.GetTensorInfo(),
                managedRecurrentToCellWeights.Map()));
    }

    if (m_QuantizedLstmParameters.m_RecurrentToOutputWeights != nullptr)
    {
        constTensors.emplace_back(ConstTensor(
                managedRecurrentToOutputWeights.GetTensorInfo(),
                managedRecurrentToOutputWeights.Map()));
    }

    // Bias tensors
    if (m_QuantizedLstmParameters.m_InputGateBias != nullptr)
    {
        constTensors.emplace_back(ConstTensor(managedInputGateBias.GetTensorInfo(),
                                              managedInputGateBias.Map()));
    }

    if (m_QuantizedLstmParameters.m_ForgetGateBias != nullptr)
    {
        constTensors.emplace_back(ConstTensor(managedForgetGateBias.GetTensorInfo(),
                                              managedForgetGateBias.Map()));
    }

    if (m_QuantizedLstmParameters.m_CellBias != nullptr)
    {
        constTensors.emplace_back(ConstTensor(managedCellBias.GetTensorInfo(),
                                              managedCellBias.Map()));
    }

    if (m_QuantizedLstmParameters.m_OutputGateBias != nullptr)
    {
        constTensors.emplace_back(ConstTensor(managedOutputGateBias.GetTensorInfo(),
                                              managedOutputGateBias.Map()));
    }


    strategy.ExecuteStrategy(this, BaseDescriptor(), constTensors, GetName());
}

} // namespace armnn
