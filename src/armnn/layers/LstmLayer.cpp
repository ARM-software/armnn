//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//
#include "LstmLayer.hpp"

#include "LayerCloneBase.hpp"

#include <armnn/TypesUtils.hpp>
#include <backendsCommon/CpuTensorHandle.hpp>
#include <backendsCommon/WorkloadFactory.hpp>

namespace armnn
{

LstmLayer::LstmLayer(const LstmDescriptor& param, const char* name)
        : LayerWithParameters(3, 4, LayerType::Lstm, param, name)
{
}

std::unique_ptr<IWorkload> LstmLayer::CreateWorkload(const Graph& graph, const IWorkloadFactory& factory) const
{
    LstmQueueDescriptor descriptor;

    // Basic parameters
    descriptor.m_InputToForgetWeights = m_BasicParameters.m_InputToForgetWeights.get();
    descriptor.m_InputToCellWeights = m_BasicParameters.m_InputToCellWeights.get();
    descriptor.m_InputToOutputWeights = m_BasicParameters.m_InputToOutputWeights.get();
    descriptor.m_RecurrentToForgetWeights = m_BasicParameters.m_RecurrentToForgetWeights.get();
    descriptor.m_RecurrentToCellWeights = m_BasicParameters.m_RecurrentToCellWeights.get();
    descriptor.m_RecurrentToOutputWeights = m_BasicParameters.m_RecurrentToOutputWeights.get();
    descriptor.m_ForgetGateBias = m_BasicParameters.m_ForgetGateBias.get();
    descriptor.m_CellBias = m_BasicParameters.m_CellBias.get();
    descriptor.m_OutputGateBias = m_BasicParameters.m_OutputGateBias.get();

    // Cifg parameters
    if (!m_Param.m_CifgEnabled)
    {
        descriptor.m_InputToInputWeights = m_CifgParameters.m_InputToInputWeights.get();
        descriptor.m_RecurrentToInputWeights = m_CifgParameters.m_RecurrentToInputWeights.get();
        descriptor.m_CellToInputWeights = m_CifgParameters.m_CellToInputWeights.get();
        descriptor.m_InputGateBias = m_CifgParameters.m_InputGateBias.get();
    }

    // Projection parameters
    if (m_Param.m_ProjectionEnabled)
    {
        descriptor.m_ProjectionWeights = m_ProjectionParameters.m_ProjectionWeights.get();
        descriptor.m_ProjectionBias    = m_ProjectionParameters.m_ProjectionBias.get();
    }

    // Peephole parameters
    if (m_Param.m_PeepholeEnabled)
    {
        descriptor.m_CellToForgetWeights = m_PeepholeParameters.m_CellToForgetWeights.get();
        descriptor.m_CellToOutputWeights = m_PeepholeParameters.m_CellToOutputWeights.get();
    }
    return factory.CreateLstm(descriptor, PrepInfoAndDesc(descriptor, graph));
}

LstmLayer* LstmLayer::Clone(Graph& graph) const
{
    auto layer = CloneBase<LstmLayer>(graph, m_Param, GetName());

    layer->m_BasicParameters.m_InputToForgetWeights = m_BasicParameters.m_InputToForgetWeights ?
            std::make_unique<ScopedCpuTensorHandle>(*m_BasicParameters.m_InputToForgetWeights)
                : nullptr;
    layer->m_BasicParameters.m_InputToCellWeights = m_BasicParameters.m_InputToCellWeights ?
            std::make_unique<ScopedCpuTensorHandle>(*m_BasicParameters.m_InputToCellWeights) : nullptr;
    layer->m_BasicParameters.m_InputToOutputWeights = m_BasicParameters.m_InputToOutputWeights ?
            std::make_unique<ScopedCpuTensorHandle>(*m_BasicParameters.m_InputToOutputWeights) : nullptr;
    layer->m_BasicParameters.m_RecurrentToForgetWeights = m_BasicParameters.m_RecurrentToForgetWeights ?
            std::make_unique<ScopedCpuTensorHandle>(*m_BasicParameters.m_RecurrentToForgetWeights) : nullptr;
    layer->m_BasicParameters.m_RecurrentToCellWeights = m_BasicParameters.m_RecurrentToCellWeights ?
            std::make_unique<ScopedCpuTensorHandle>(*m_BasicParameters.m_RecurrentToCellWeights) : nullptr;
    layer->m_BasicParameters.m_RecurrentToOutputWeights = m_BasicParameters.m_RecurrentToOutputWeights ?
            std::make_unique<ScopedCpuTensorHandle>(*m_BasicParameters.m_RecurrentToOutputWeights) : nullptr;
    layer->m_BasicParameters.m_ForgetGateBias = m_BasicParameters.m_ForgetGateBias ?
            std::make_unique<ScopedCpuTensorHandle>(*m_BasicParameters.m_ForgetGateBias) : nullptr;
    layer->m_BasicParameters.m_CellBias = m_BasicParameters.m_CellBias ?
            std::make_unique<ScopedCpuTensorHandle>(*m_BasicParameters.m_CellBias) : nullptr;
    layer->m_BasicParameters.m_OutputGateBias = m_BasicParameters.m_OutputGateBias ?
            std::make_unique<ScopedCpuTensorHandle>(*m_BasicParameters.m_OutputGateBias) : nullptr;

    if (!m_Param.m_CifgEnabled)
    {
        layer->m_CifgParameters.m_InputToInputWeights = m_CifgParameters.m_InputToInputWeights ?
                std::make_unique<ScopedCpuTensorHandle>(*m_CifgParameters.m_InputToInputWeights) : nullptr;
        layer->m_CifgParameters.m_RecurrentToInputWeights = m_CifgParameters.m_RecurrentToInputWeights ?
                std::make_unique<ScopedCpuTensorHandle>(*m_CifgParameters.m_RecurrentToInputWeights) : nullptr;
        layer->m_CifgParameters.m_CellToInputWeights = m_CifgParameters.m_CellToInputWeights ?
                std::make_unique<ScopedCpuTensorHandle>(*m_CifgParameters.m_CellToInputWeights) : nullptr;
        layer->m_CifgParameters.m_InputGateBias = m_CifgParameters.m_InputGateBias ?
                std::make_unique<ScopedCpuTensorHandle>(*m_CifgParameters.m_InputGateBias) : nullptr;
    }

    if (m_Param.m_ProjectionEnabled)
    {
        layer->m_ProjectionParameters.m_ProjectionWeights = m_ProjectionParameters.m_ProjectionWeights ?
               std::make_unique<ScopedCpuTensorHandle>(*m_ProjectionParameters.m_ProjectionWeights) : nullptr;
        layer->m_ProjectionParameters.m_ProjectionBias = m_ProjectionParameters.m_ProjectionBias ?
               std::make_unique<ScopedCpuTensorHandle>(*m_ProjectionParameters.m_ProjectionBias) : nullptr;
    }

    if (m_Param.m_PeepholeEnabled)
    {
        layer->m_PeepholeParameters.m_CellToForgetWeights = m_PeepholeParameters.m_CellToForgetWeights ?
               std::make_unique<ScopedCpuTensorHandle>(*m_PeepholeParameters.m_CellToForgetWeights) : nullptr;
        layer->m_PeepholeParameters.m_CellToOutputWeights = m_PeepholeParameters.m_CellToOutputWeights ?
               std::make_unique<ScopedCpuTensorHandle>(*m_PeepholeParameters.m_CellToOutputWeights) : nullptr;
    }

    return std::move(layer);
}

std::vector<TensorShape> LstmLayer::InferOutputShapes(const std::vector<TensorShape>& inputShapes) const
{
    BOOST_ASSERT(inputShapes.size() == 3);

    // Get input values for validation
    unsigned int batchSize = inputShapes[0][0];
    unsigned int outputSize = inputShapes[1][1];
    unsigned int numUnits = inputShapes[2][1];

    std::vector<TensorShape> outShapes;
    outShapes.push_back(TensorShape({batchSize, numUnits * (m_Param.m_CifgEnabled ? 3 : 4)}));
    outShapes.push_back(TensorShape({batchSize, outputSize}));
    outShapes.push_back(TensorShape({batchSize, numUnits}));
    outShapes.push_back(TensorShape({batchSize, outputSize}));

    return outShapes;
}

void LstmLayer::ValidateTensorShapesFromInputs()
{
    VerifyLayerConnections(3, CHECK_LOCATION());

    auto inferredShapes = InferOutputShapes( {
        GetInputSlot(0).GetConnection()->GetTensorInfo().GetShape(),
        GetInputSlot(1).GetConnection()->GetTensorInfo().GetShape(),
        GetInputSlot(2).GetConnection()->GetTensorInfo().GetShape()}
    );

    BOOST_ASSERT(inferredShapes.size() == 4);

    // Check if the weights are nullptr
    BOOST_ASSERT_MSG(m_BasicParameters.m_InputToForgetWeights != nullptr,
                     "LstmLayer: m_BasicParameters.m_InputToForgetWeights should not be null.");
    BOOST_ASSERT_MSG(m_BasicParameters.m_InputToCellWeights != nullptr,
                     "LstmLayer: m_BasicParameters.m_InputToCellWeights should not be null.");
    BOOST_ASSERT_MSG(m_BasicParameters.m_InputToOutputWeights != nullptr,
                     "LstmLayer: m_BasicParameters.m_InputToOutputWeights should not be null.");
    BOOST_ASSERT_MSG(m_BasicParameters.m_RecurrentToForgetWeights != nullptr,
                     "LstmLayer: m_BasicParameters.m_RecurrentToForgetWeights should not be null.");
    BOOST_ASSERT_MSG(m_BasicParameters.m_RecurrentToCellWeights != nullptr,
                     "LstmLayer: m_BasicParameters.m_RecurrentToCellWeights should not be null.");
    BOOST_ASSERT_MSG(m_BasicParameters.m_RecurrentToOutputWeights != nullptr,
                     "LstmLayer: m_BasicParameters.m_RecurrentToOutputWeights should not be null.");
    BOOST_ASSERT_MSG(m_BasicParameters.m_ForgetGateBias != nullptr,
                     "LstmLayer: m_BasicParameters.m_ForgetGateBias should not be null.");
    BOOST_ASSERT_MSG(m_BasicParameters.m_CellBias != nullptr,
                     "LstmLayer: m_BasicParameters.m_CellBias should not be null.");
    BOOST_ASSERT_MSG(m_BasicParameters.m_OutputGateBias != nullptr,
                     "LstmLayer: m_BasicParameters.m_OutputGateBias should not be null.");

    if (!m_Param.m_CifgEnabled)
    {
        BOOST_ASSERT_MSG(m_CifgParameters.m_InputToInputWeights != nullptr,
                         "LstmLayer: m_CifgParameters.m_InputToInputWeights should not be null.");
        BOOST_ASSERT_MSG(m_CifgParameters.m_RecurrentToInputWeights != nullptr,
                         "LstmLayer: m_CifgParameters.m_RecurrentToInputWeights should not be null.");
        BOOST_ASSERT_MSG(m_CifgParameters.m_InputGateBias != nullptr,
                         "LstmLayer: m_CifgParameters.m_InputGateBias should not be null.");

        ConditionalThrowIfNotEqual<LayerValidationException>(
                "LstmLayer: TensorShape set on OutputSlot[0] does not match the inferred shape.",
                GetOutputSlot(0).GetTensorInfo().GetShape(),
                inferredShapes[0]);
    }
    else
    {
        BOOST_ASSERT_MSG(m_CifgParameters.m_InputToInputWeights == nullptr,
            "LstmLayer: m_CifgParameters.m_InputToInputWeights should not have a value when CIFG is enabled.");
        BOOST_ASSERT_MSG(m_CifgParameters.m_RecurrentToInputWeights == nullptr,
            "LstmLayer: m_CifgParameters.m_RecurrentToInputWeights should not have a value when CIFG is enabled.");
        BOOST_ASSERT_MSG(m_CifgParameters.m_CellToInputWeights == nullptr,
             "LstmLayer: m_CifgParameters.m_CellToInputWeights should not have a value when CIFG is enabled.");
        BOOST_ASSERT_MSG(m_CifgParameters.m_InputGateBias == nullptr,
            "LstmLayer: m_CifgParameters.m_InputGateBias should not have a value when CIFG is enabled.");

        ConditionalThrowIfNotEqual<LayerValidationException>(
                "LstmLayer: TensorShape set on OutputSlot[0] does not match the inferred shape.",
                GetOutputSlot(0).GetTensorInfo().GetShape(),
                inferredShapes[0]);
    }

    if (m_Param.m_ProjectionEnabled)
    {
        BOOST_ASSERT_MSG(m_ProjectionParameters.m_ProjectionWeights != nullptr,
                         "LstmLayer: m_ProjectionParameters.m_ProjectionWeights should not be null.");
    }

    if (m_Param.m_PeepholeEnabled)
    {
        BOOST_ASSERT_MSG(m_PeepholeParameters.m_CellToForgetWeights != nullptr,
                         "LstmLayer: m_PeepholeParameters.m_CellToForgetWeights should not be null.");
        BOOST_ASSERT_MSG(m_PeepholeParameters.m_CellToOutputWeights != nullptr,
                         "LstmLayer: m_PeepholeParameters.m_CellToOutputWeights should not be null.");
    }

    ConditionalThrowIfNotEqual<LayerValidationException>(
            "LstmLayer: TensorShape set on OutputSlot[1] does not match the inferred shape.",
            GetOutputSlot(1).GetTensorInfo().GetShape(),
            inferredShapes[1]);
    ConditionalThrowIfNotEqual<LayerValidationException>(
            "LstmLayer: TensorShape set on OutputSlot[2] does not match the inferred shape.",
            GetOutputSlot(2).GetTensorInfo().GetShape(),
            inferredShapes[2]);
    ConditionalThrowIfNotEqual<LayerValidationException>(
            "LstmLayer: TensorShape set on OutputSlot[3] does not match the inferred shape.",
            GetOutputSlot(3).GetTensorInfo().GetShape(),
            inferredShapes[3]);
}

Layer::ConstantTensors LstmLayer::GetConstantTensorsByRef()
{
    return {m_BasicParameters.m_InputToForgetWeights,
            m_BasicParameters.m_InputToCellWeights,
            m_BasicParameters.m_InputToOutputWeights,
            m_BasicParameters.m_RecurrentToForgetWeights,
            m_BasicParameters.m_RecurrentToCellWeights,
            m_BasicParameters.m_RecurrentToOutputWeights,
            m_BasicParameters.m_ForgetGateBias,
            m_BasicParameters.m_CellBias,
            m_BasicParameters.m_OutputGateBias,

            // Cifg parameters
            m_CifgParameters.m_InputToInputWeights,
            m_CifgParameters.m_RecurrentToInputWeights,
            m_CifgParameters.m_CellToInputWeights,
            m_CifgParameters.m_InputGateBias,

            // Projection parameters
            m_ProjectionParameters.m_ProjectionWeights,
            m_ProjectionParameters.m_ProjectionBias,

            // Peephole parameters
            m_PeepholeParameters.m_CellToForgetWeights,
            m_PeepholeParameters.m_CellToOutputWeights};
}

void LstmLayer::Accept(ILayerVisitor& visitor) const
{
    LstmInputParams inputParams;
    if (m_CifgParameters.m_InputToInputWeights != nullptr)
    {
        ConstTensor inputToInputWeightsTensor(m_CifgParameters.m_InputToInputWeights->GetTensorInfo(),
                                              m_CifgParameters.m_InputToInputWeights->GetConstTensor<void*>());
        inputParams.m_InputToInputWeights = &inputToInputWeightsTensor;
    }
    if (m_BasicParameters.m_InputToForgetWeights != nullptr)
    {
        ConstTensor inputToForgetWeightsTensor(m_BasicParameters.m_InputToForgetWeights->GetTensorInfo(),
                                               m_BasicParameters.m_InputToForgetWeights->GetConstTensor<void*>());
        inputParams.m_InputToForgetWeights = &inputToForgetWeightsTensor;
    }
    if (m_BasicParameters.m_InputToCellWeights != nullptr)
    {
        ConstTensor inputToCellWeightsTensor(m_BasicParameters.m_InputToCellWeights->GetTensorInfo(),
                                             m_BasicParameters.m_InputToCellWeights->GetConstTensor<void*>());
        inputParams.m_InputToCellWeights = &inputToCellWeightsTensor;
    }
    if (m_BasicParameters.m_InputToOutputWeights != nullptr)
    {
        ConstTensor inputToOutputWeightsTensor(m_BasicParameters.m_InputToOutputWeights->GetTensorInfo(),
                                               m_BasicParameters.m_InputToOutputWeights->GetConstTensor<void*>());
        inputParams.m_InputToOutputWeights = &inputToOutputWeightsTensor;
    }
    if (m_CifgParameters.m_RecurrentToInputWeights != nullptr)
    {
        ConstTensor recurrentToInputWeightsTensor(
                m_CifgParameters.m_RecurrentToInputWeights->GetTensorInfo(),
                m_CifgParameters.m_RecurrentToInputWeights->GetConstTensor<void*>());
        inputParams.m_RecurrentToInputWeights = &recurrentToInputWeightsTensor;
    }
    if (m_BasicParameters.m_RecurrentToForgetWeights != nullptr)
    {
        ConstTensor recurrentToForgetWeightsTensor(
                m_BasicParameters.m_RecurrentToForgetWeights->GetTensorInfo(),
                m_BasicParameters.m_RecurrentToForgetWeights->GetConstTensor<void*>());
        inputParams.m_RecurrentToForgetWeights = &recurrentToForgetWeightsTensor;
    }
    if (m_BasicParameters.m_RecurrentToCellWeights != nullptr)
    {
        ConstTensor recurrentToCellWeightsTensor(
                m_BasicParameters.m_RecurrentToCellWeights->GetTensorInfo(),
                m_BasicParameters.m_RecurrentToCellWeights->GetConstTensor<void*>());
        inputParams.m_RecurrentToCellWeights = &recurrentToCellWeightsTensor;
    }
    if (m_BasicParameters.m_RecurrentToOutputWeights != nullptr)
    {
        ConstTensor recurrentToOutputWeightsTensor(
                m_BasicParameters.m_RecurrentToOutputWeights->GetTensorInfo(),
                m_BasicParameters.m_RecurrentToOutputWeights->GetConstTensor<void*>());
        inputParams.m_RecurrentToOutputWeights = &recurrentToOutputWeightsTensor;
    }
    if (m_CifgParameters.m_CellToInputWeights != nullptr)
    {
        ConstTensor cellToInputWeightsTensor(m_CifgParameters.m_CellToInputWeights->GetTensorInfo(),
                                             m_CifgParameters.m_CellToInputWeights->GetConstTensor<void*>());
        inputParams.m_CellToInputWeights = &cellToInputWeightsTensor;
    }
    if (m_PeepholeParameters.m_CellToForgetWeights != nullptr)
    {
        ConstTensor cellToForgetWeightsTensor(m_PeepholeParameters.m_CellToForgetWeights->GetTensorInfo(),
                                              m_PeepholeParameters.m_CellToForgetWeights->GetConstTensor<void*>());
        inputParams.m_CellToForgetWeights = &cellToForgetWeightsTensor;
    }
    if (m_PeepholeParameters.m_CellToOutputWeights != nullptr)
    {
        ConstTensor cellToOutputWeightsTensor(m_PeepholeParameters.m_CellToOutputWeights->GetTensorInfo(),
                                              m_PeepholeParameters.m_CellToOutputWeights->GetConstTensor<void*>());
        inputParams.m_CellToOutputWeights = &cellToOutputWeightsTensor;
    }
    if (m_CifgParameters.m_InputGateBias != nullptr)
    {
        ConstTensor inputGateBiasTensor(m_CifgParameters.m_InputGateBias->GetTensorInfo(),
                                        m_CifgParameters.m_InputGateBias->GetConstTensor<void*>());
        inputParams.m_InputGateBias = &inputGateBiasTensor;
    }
    if (m_BasicParameters.m_ForgetGateBias != nullptr)
    {
        ConstTensor forgetGateBiasTensor(m_BasicParameters.m_ForgetGateBias->GetTensorInfo(),
                                         m_BasicParameters.m_ForgetGateBias->GetConstTensor<void*>());
        inputParams.m_ForgetGateBias = &forgetGateBiasTensor;
    }
    if (m_BasicParameters.m_CellBias != nullptr)
    {
        ConstTensor cellBiasTensor(m_BasicParameters.m_CellBias->GetTensorInfo(),
                                   m_BasicParameters.m_CellBias->GetConstTensor<void*>());
        inputParams.m_CellBias = &cellBiasTensor;
    }
    if (m_BasicParameters.m_OutputGateBias != nullptr)
    {
        ConstTensor outputGateBias(m_BasicParameters.m_OutputGateBias->GetTensorInfo(),
                                   m_BasicParameters.m_OutputGateBias->GetConstTensor<void*>());
        inputParams.m_OutputGateBias = &outputGateBias;
    }
    if (m_ProjectionParameters.m_ProjectionWeights != nullptr)
    {
        ConstTensor projectionWeightsTensor(m_ProjectionParameters.m_ProjectionWeights->GetTensorInfo(),
                                            m_ProjectionParameters.m_ProjectionWeights->GetConstTensor<void*>());
        inputParams.m_ProjectionWeights = &projectionWeightsTensor;
    }
    if (m_ProjectionParameters.m_ProjectionBias != nullptr)
    {
        ConstTensor projectionBiasTensor(m_ProjectionParameters.m_ProjectionBias->GetTensorInfo(),
                                         m_ProjectionParameters.m_ProjectionBias->GetConstTensor<void*>());
        inputParams.m_ProjectionBias = &projectionBiasTensor;
    }

    visitor.VisitLstmLayer(this, GetParameters(), inputParams, GetName());
}

} // namespace armnn
