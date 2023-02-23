//
// Copyright © 2020 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//
#include "QLstmLayer.hpp"

#include "LayerCloneBase.hpp"

#include <armnn/LstmParams.hpp>
#include <armnn/TypesUtils.hpp>
#include <armnn/backends/TensorHandle.hpp>
#include <armnn/backends/WorkloadFactory.hpp>

namespace armnn
{

QLstmLayer::QLstmLayer(const QLstmDescriptor& param, const char* name)
        : LayerWithParameters(3, 3, LayerType::QLstm, param, name)
{
}

std::unique_ptr<IWorkload> QLstmLayer::CreateWorkload(const IWorkloadFactory& factory) const
{
    QLstmQueueDescriptor descriptor;

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

    // CIFG parameters
    if (!m_Param.m_CifgEnabled)
    {
        descriptor.m_InputToInputWeights     = m_CifgParameters.m_InputToInputWeights.get();
        descriptor.m_RecurrentToInputWeights = m_CifgParameters.m_RecurrentToInputWeights.get();
        descriptor.m_InputGateBias           = m_CifgParameters.m_InputGateBias.get();
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
        if (!m_Param.m_CifgEnabled)
        {
            descriptor.m_CellToInputWeights = m_PeepholeParameters.m_CellToInputWeights.get();
        }

        descriptor.m_CellToForgetWeights = m_PeepholeParameters.m_CellToForgetWeights.get();
        descriptor.m_CellToOutputWeights = m_PeepholeParameters.m_CellToOutputWeights.get();
    }

    // Layer normalisation parameters
    if(m_Param.m_LayerNormEnabled)
    {
        if (!m_Param.m_CifgEnabled)
        {
            descriptor.m_InputLayerNormWeights = m_LayerNormParameters.m_InputLayerNormWeights.get();
        }
        descriptor.m_ForgetLayerNormWeights = m_LayerNormParameters.m_ForgetLayerNormWeights.get();
        descriptor.m_CellLayerNormWeights   = m_LayerNormParameters.m_CellLayerNormWeights.get();
        descriptor.m_OutputLayerNormWeights = m_LayerNormParameters.m_OutputLayerNormWeights.get();
    }

    SetAdditionalInfo(descriptor);

    return factory.CreateWorkload(LayerType::QLstm, descriptor, PrepInfoAndDesc(descriptor));
}

QLstmLayer* QLstmLayer::Clone(Graph& graph) const
{
    auto layer = CloneBase<QLstmLayer>(graph, m_Param, GetName());

    layer->m_BasicParameters.m_InputToForgetWeights = m_BasicParameters.m_InputToForgetWeights ?
            m_BasicParameters.m_InputToForgetWeights : nullptr;
    layer->m_BasicParameters.m_InputToCellWeights = m_BasicParameters.m_InputToCellWeights ?
            m_BasicParameters.m_InputToCellWeights : nullptr;
    layer->m_BasicParameters.m_InputToOutputWeights = m_BasicParameters.m_InputToOutputWeights ?
            m_BasicParameters.m_InputToOutputWeights : nullptr;
    layer->m_BasicParameters.m_RecurrentToForgetWeights = m_BasicParameters.m_RecurrentToForgetWeights ?
            m_BasicParameters.m_RecurrentToForgetWeights : nullptr;
    layer->m_BasicParameters.m_RecurrentToCellWeights = m_BasicParameters.m_RecurrentToCellWeights ?
            m_BasicParameters.m_RecurrentToCellWeights : nullptr;
    layer->m_BasicParameters.m_RecurrentToOutputWeights = m_BasicParameters.m_RecurrentToOutputWeights ?
            m_BasicParameters.m_RecurrentToOutputWeights : nullptr;
    layer->m_BasicParameters.m_ForgetGateBias = m_BasicParameters.m_ForgetGateBias ?
            m_BasicParameters.m_ForgetGateBias : nullptr;
    layer->m_BasicParameters.m_CellBias = m_BasicParameters.m_CellBias ?
            m_BasicParameters.m_CellBias : nullptr;
    layer->m_BasicParameters.m_OutputGateBias = m_BasicParameters.m_OutputGateBias ?
            m_BasicParameters.m_OutputGateBias : nullptr;

    if (!m_Param.m_CifgEnabled)
    {
        layer->m_CifgParameters.m_InputToInputWeights = m_CifgParameters.m_InputToInputWeights ?
                m_CifgParameters.m_InputToInputWeights : nullptr;
        layer->m_CifgParameters.m_RecurrentToInputWeights = m_CifgParameters.m_RecurrentToInputWeights ?
                m_CifgParameters.m_RecurrentToInputWeights : nullptr;
        layer->m_CifgParameters.m_InputGateBias = m_CifgParameters.m_InputGateBias ?
                m_CifgParameters.m_InputGateBias : nullptr;
    }

    if (m_Param.m_ProjectionEnabled)
    {
        layer->m_ProjectionParameters.m_ProjectionWeights = m_ProjectionParameters.m_ProjectionWeights ?
                m_ProjectionParameters.m_ProjectionWeights : nullptr;
        layer->m_ProjectionParameters.m_ProjectionBias = m_ProjectionParameters.m_ProjectionBias ?
                m_ProjectionParameters.m_ProjectionBias : nullptr;
    }

    if (m_Param.m_PeepholeEnabled)
    {
        if (!m_Param.m_CifgEnabled) {
            layer->m_PeepholeParameters.m_CellToInputWeights = m_PeepholeParameters.m_CellToInputWeights ?
                    m_PeepholeParameters.m_CellToInputWeights : nullptr;
        }

        layer->m_PeepholeParameters.m_CellToForgetWeights = m_PeepholeParameters.m_CellToForgetWeights ?
                m_PeepholeParameters.m_CellToForgetWeights : nullptr;
        layer->m_PeepholeParameters.m_CellToOutputWeights = m_PeepholeParameters.m_CellToOutputWeights ?
                m_PeepholeParameters.m_CellToOutputWeights : nullptr;
    }

    if (m_Param.m_LayerNormEnabled)
    {
        if (!m_Param.m_CifgEnabled) {
            layer->m_LayerNormParameters.m_InputLayerNormWeights = m_LayerNormParameters.m_InputLayerNormWeights ?
                    m_LayerNormParameters.m_InputLayerNormWeights : nullptr;
        }

        layer->m_LayerNormParameters.m_ForgetLayerNormWeights = m_LayerNormParameters.m_ForgetLayerNormWeights ?
                m_LayerNormParameters.m_ForgetLayerNormWeights : nullptr;
        layer->m_LayerNormParameters.m_CellLayerNormWeights = m_LayerNormParameters.m_CellLayerNormWeights ?
                m_LayerNormParameters.m_CellLayerNormWeights : nullptr;
        layer->m_LayerNormParameters.m_OutputLayerNormWeights = m_LayerNormParameters.m_OutputLayerNormWeights ?
                m_LayerNormParameters.m_OutputLayerNormWeights : nullptr;
    }

    return std::move(layer);
}

std::vector<TensorShape> QLstmLayer::InferOutputShapes(const std::vector<TensorShape>& inputShapes) const
{
    ARMNN_ASSERT(inputShapes.size() == 3);

    // Get input values for validation
    unsigned int batchSize = inputShapes[0][0];
    unsigned int outputSize = inputShapes[1][1];
    unsigned int numUnits = inputShapes[2][1];

    std::vector<TensorShape> outShapes;
    outShapes.push_back(TensorShape({ batchSize, outputSize })); // outputStateOut
    outShapes.push_back(TensorShape({ batchSize, numUnits })); // cellStateOut
    outShapes.push_back(TensorShape({ batchSize, outputSize })); // output

    return outShapes;
}

void QLstmLayer::ValidateTensorShapesFromInputs()
{
    VerifyLayerConnections(3, CHECK_LOCATION());

    const TensorShape& outputShape = GetOutputSlot(0).GetTensorInfo().GetShape();

    VerifyShapeInferenceType(outputShape, m_ShapeInferenceMethod);

    auto inferredShapes = InferOutputShapes(
    {
        GetInputSlot(0).GetConnection()->GetTensorInfo().GetShape(), // input
        GetInputSlot(1).GetConnection()->GetTensorInfo().GetShape(), // previousOutputIn
        GetInputSlot(2).GetConnection()->GetTensorInfo().GetShape()  // previousCellStateIn
    });

    ARMNN_ASSERT(inferredShapes.size() == 3);

    // Check if the weights are nullptr for basic params
    ARMNN_ASSERT_MSG(m_BasicParameters.m_InputToForgetWeights != nullptr,
            "QLstmLayer: m_BasicParameters.m_InputToForgetWeights should not be null.");
    ARMNN_ASSERT_MSG(m_BasicParameters.m_InputToCellWeights != nullptr,
            "QLstmLayer: m_BasicParameters.m_InputToCellWeights should not be null.");
    ARMNN_ASSERT_MSG(m_BasicParameters.m_InputToOutputWeights != nullptr,
            "QLstmLayer: m_BasicParameters.m_InputToOutputWeights should not be null.");
    ARMNN_ASSERT_MSG(m_BasicParameters.m_RecurrentToForgetWeights != nullptr,
            "QLstmLayer: m_BasicParameters.m_RecurrentToForgetWeights should not be null.");
    ARMNN_ASSERT_MSG(m_BasicParameters.m_RecurrentToCellWeights != nullptr,
            "QLstmLayer: m_BasicParameters.m_RecurrentToCellWeights should not be null.");
    ARMNN_ASSERT_MSG(m_BasicParameters.m_RecurrentToOutputWeights != nullptr,
            "QLstmLayer: m_BasicParameters.m_RecurrentToOutputWeights should not be null.");
    ARMNN_ASSERT_MSG(m_BasicParameters.m_ForgetGateBias != nullptr,
            "QLstmLayer: m_BasicParameters.m_ForgetGateBias should not be null.");
    ARMNN_ASSERT_MSG(m_BasicParameters.m_CellBias != nullptr,
            "QLstmLayer: m_BasicParameters.m_CellBias should not be null.");
    ARMNN_ASSERT_MSG(m_BasicParameters.m_OutputGateBias != nullptr,
            "QLstmLayer: m_BasicParameters.m_OutputGateBias should not be null.");

    if (!m_Param.m_CifgEnabled)
    {
        ARMNN_ASSERT_MSG(m_CifgParameters.m_InputToInputWeights != nullptr,
                "QLstmLayer: m_CifgParameters.m_InputToInputWeights should not be null.");
        ARMNN_ASSERT_MSG(m_CifgParameters.m_RecurrentToInputWeights != nullptr,
                "QLstmLayer: m_CifgParameters.m_RecurrentToInputWeights should not be null.");
        ARMNN_ASSERT_MSG(m_CifgParameters.m_InputGateBias != nullptr,
                "QLstmLayer: m_CifgParameters.m_InputGateBias should not be null.");

        ValidateAndCopyShape(outputShape, inferredShapes[0], m_ShapeInferenceMethod, "QLstmLayer");
    }
    else
    {
        ARMNN_ASSERT_MSG(m_CifgParameters.m_InputToInputWeights == nullptr,
                "QLstmLayer: m_CifgParameters.m_InputToInputWeights should not have a value when CIFG is enabled.");
        ARMNN_ASSERT_MSG(m_CifgParameters.m_RecurrentToInputWeights == nullptr,
                "QLstmLayer: m_CifgParameters.m_RecurrentToInputWeights should "
                             "not have a value when CIFG is enabled.");
        ARMNN_ASSERT_MSG(m_CifgParameters.m_InputGateBias == nullptr,
                "QLstmLayer: m_CifgParameters.m_InputGateBias should not have a value when CIFG is enabled.");

        ValidateAndCopyShape(outputShape, inferredShapes[0], m_ShapeInferenceMethod, "QLstmLayer");
    }

    if (m_Param.m_ProjectionEnabled)
    {
        ARMNN_ASSERT_MSG(m_ProjectionParameters.m_ProjectionWeights != nullptr,
                         "QLstmLayer: m_ProjectionParameters.m_ProjectionWeights should not be null.");
    }

    if (m_Param.m_PeepholeEnabled)
    {
        if (!m_Param.m_CifgEnabled) {
            ARMNN_ASSERT_MSG(m_PeepholeParameters.m_CellToInputWeights != nullptr,
                    "QLstmLayer: m_PeepholeParameters.m_CellToInputWeights should not be null "
                    "when Peephole is enabled and CIFG is disabled.");
        }

        ARMNN_ASSERT_MSG(m_PeepholeParameters.m_CellToForgetWeights != nullptr,
                         "QLstmLayer: m_PeepholeParameters.m_CellToForgetWeights should not be null.");
        ARMNN_ASSERT_MSG(m_PeepholeParameters.m_CellToOutputWeights != nullptr,
                         "QLstmLayer: m_PeepholeParameters.m_CellToOutputWeights should not be null.");
    }

    ValidateAndCopyShape(
            GetOutputSlot(1).GetTensorInfo().GetShape(), inferredShapes[1], m_ShapeInferenceMethod, "QLstmLayer", 1);
    ValidateAndCopyShape(
            GetOutputSlot(2).GetTensorInfo().GetShape(), inferredShapes[2], m_ShapeInferenceMethod, "QLstmLayer", 2);

    if (m_Param.m_LayerNormEnabled)
    {
        if(!m_Param.m_CifgEnabled)
        {
            ARMNN_ASSERT_MSG(m_LayerNormParameters.m_InputLayerNormWeights != nullptr,
                             "QLstmLayer: m_LayerNormParameters.m_InputLayerNormWeights should not be null.");
        }
        ARMNN_ASSERT_MSG(m_LayerNormParameters.m_ForgetLayerNormWeights != nullptr,
                         "QLstmLayer: m_LayerNormParameters.m_ForgetLayerNormWeights should not be null.");
        ARMNN_ASSERT_MSG(m_LayerNormParameters.m_CellLayerNormWeights != nullptr,
                         "QLstmLayer: m_LayerNormParameters.m_CellLayerNormWeights should not be null.");
        ARMNN_ASSERT_MSG(m_LayerNormParameters.m_OutputLayerNormWeights != nullptr,
                         "QLstmLayer: m_LayerNormParameters.m_UutputLayerNormWeights should not be null.");
    }
}

Layer::ImmutableConstantTensors QLstmLayer::GetConstantTensorsByRef() const
{
    // For API stability DO NOT ALTER order and add new members to the end of vector
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
            m_CifgParameters.m_InputGateBias,

            // Projection parameters
            m_ProjectionParameters.m_ProjectionWeights,
            m_ProjectionParameters.m_ProjectionBias,

            // Peephole parameters
            m_PeepholeParameters.m_CellToInputWeights,
            m_PeepholeParameters.m_CellToForgetWeights,
            m_PeepholeParameters.m_CellToOutputWeights,

            // Layer normalisation parameters
            m_LayerNormParameters.m_InputLayerNormWeights,
            m_LayerNormParameters.m_ForgetLayerNormWeights,
            m_LayerNormParameters.m_CellLayerNormWeights,
            m_LayerNormParameters.m_OutputLayerNormWeights};
}


void QLstmLayer::ExecuteStrategy(IStrategy& strategy) const
{
    std::vector<ConstTensor> constTensors;
    ManagedConstTensorHandle managedInputToForgetWeights(m_BasicParameters.m_InputToForgetWeights);
    ManagedConstTensorHandle managedInputToCellWeights(m_BasicParameters.m_InputToCellWeights);
    ManagedConstTensorHandle managedInputToOutputWeights(m_BasicParameters.m_InputToOutputWeights);
    ManagedConstTensorHandle managedRecurrentToForgetWeights(m_BasicParameters.m_RecurrentToForgetWeights);
    ManagedConstTensorHandle managedRecurrentToCellWeights(m_BasicParameters.m_RecurrentToCellWeights);
    ManagedConstTensorHandle managedRecurrentToOutputWeights(m_BasicParameters.m_RecurrentToOutputWeights);
    ManagedConstTensorHandle managedForgetGateBias(m_BasicParameters.m_ForgetGateBias);
    ManagedConstTensorHandle managedCellBias(m_BasicParameters.m_CellBias);
    ManagedConstTensorHandle managedOutputGateBias(m_BasicParameters.m_OutputGateBias);

    // Cifg parameters
    ManagedConstTensorHandle managedInputToInputWeights(m_CifgParameters.m_InputToInputWeights);
    ManagedConstTensorHandle managedRecurrentToInputWeights(m_CifgParameters.m_RecurrentToInputWeights);
    ManagedConstTensorHandle managedInputGateBias(m_CifgParameters.m_InputGateBias);

    // Projection parameters
    ManagedConstTensorHandle managedProjectionWeights(m_ProjectionParameters.m_ProjectionWeights);
    ManagedConstTensorHandle managedProjectionBias(m_ProjectionParameters.m_ProjectionBias);

    // Peephole parameters
    ManagedConstTensorHandle managedCellToInputWeights(m_PeepholeParameters.m_CellToInputWeights);
    ManagedConstTensorHandle managedCellToForgetWeights(m_PeepholeParameters.m_CellToForgetWeights);
    ManagedConstTensorHandle managedCellToOutputWeights(m_PeepholeParameters.m_CellToOutputWeights);

    // Layer normalisation parameters
    ManagedConstTensorHandle managedInputLayerNormWeights(m_LayerNormParameters.m_InputLayerNormWeights);
    ManagedConstTensorHandle managedForgetLayerNormWeights(m_LayerNormParameters.m_ForgetLayerNormWeights);
    ManagedConstTensorHandle managedCellLayerNormWeights(m_LayerNormParameters.m_CellLayerNormWeights);
    ManagedConstTensorHandle managedOutputLayerNormWeights(m_LayerNormParameters.m_OutputLayerNormWeights);

    // First add mandatory/basic parameters
    if (m_BasicParameters.m_InputToForgetWeights != nullptr)
    {
        constTensors.emplace_back(ConstTensor(managedInputToForgetWeights.GetTensorInfo(),
                                              managedInputToForgetWeights.Map()));
    }
    if (m_BasicParameters.m_InputToCellWeights != nullptr)
    {
        constTensors.emplace_back(ConstTensor(managedInputToCellWeights.GetTensorInfo(),
                                              managedInputToCellWeights.Map()));
    }
    if (m_BasicParameters.m_InputToOutputWeights != nullptr)
    {
        constTensors.emplace_back(ConstTensor(managedInputToOutputWeights.GetTensorInfo(),
                                              managedInputToOutputWeights.Map()));
    }
    if (m_BasicParameters.m_RecurrentToForgetWeights != nullptr)
    {
        constTensors.emplace_back(ConstTensor(
                managedRecurrentToForgetWeights.GetTensorInfo(),
                managedRecurrentToForgetWeights.Map()));
    }
    if (m_BasicParameters.m_RecurrentToCellWeights != nullptr)
    {
        constTensors.emplace_back(ConstTensor(
                managedRecurrentToCellWeights.GetTensorInfo(),
                managedRecurrentToCellWeights.Map()));
    }
    if (m_BasicParameters.m_RecurrentToOutputWeights != nullptr)
    {
        constTensors.emplace_back(ConstTensor(
                managedRecurrentToOutputWeights.GetTensorInfo(),
                managedRecurrentToOutputWeights.Map()));
    }
    if (m_BasicParameters.m_ForgetGateBias != nullptr)
    {
        constTensors.emplace_back(ConstTensor(managedForgetGateBias.GetTensorInfo(),
                                              managedForgetGateBias.Map()));
    }
    if (m_BasicParameters.m_CellBias != nullptr)
    {
        constTensors.emplace_back(ConstTensor(managedCellBias.GetTensorInfo(),
                                              managedCellBias.Map()));
    }
    if (m_BasicParameters.m_OutputGateBias != nullptr)
    {
        constTensors.emplace_back(ConstTensor(managedOutputGateBias.GetTensorInfo(),
                                              managedOutputGateBias.Map()));
    }

    // Add cifig parameters
    if (m_CifgParameters.m_InputToInputWeights != nullptr)
    {
        constTensors.emplace_back(ConstTensor(managedInputToInputWeights.GetTensorInfo(),
                                              managedInputToInputWeights.Map()));
    }
    if (m_CifgParameters.m_RecurrentToInputWeights != nullptr)
    {
        constTensors.emplace_back(ConstTensor(
                managedRecurrentToInputWeights.GetTensorInfo(),
                managedRecurrentToInputWeights.Map()));
    }
    if (m_CifgParameters.m_InputGateBias != nullptr)
    {
        constTensors.emplace_back(ConstTensor(managedInputGateBias.GetTensorInfo(),
                                              managedInputGateBias.Map()));
    }

    // Add peephole parameters
    if (m_PeepholeParameters.m_CellToInputWeights != nullptr)
    {
        constTensors.emplace_back(ConstTensor(managedCellToInputWeights.GetTensorInfo(),
                                              managedCellToInputWeights.Map()));
    }
    if (m_PeepholeParameters.m_CellToForgetWeights != nullptr)
    {
        constTensors.emplace_back(ConstTensor(managedCellToForgetWeights.GetTensorInfo(),
                                              managedCellToForgetWeights.Map()));
    }
    if (m_PeepholeParameters.m_CellToOutputWeights != nullptr)
    {
        constTensors.emplace_back(ConstTensor(managedCellToOutputWeights.GetTensorInfo(),
                                              managedCellToOutputWeights.Map()));
    }

    // Add projection parameters
    if (m_ProjectionParameters.m_ProjectionWeights != nullptr)
    {
        constTensors.emplace_back(ConstTensor(managedProjectionWeights.GetTensorInfo(),
                                              managedProjectionWeights.Map()));
    }
    if (m_ProjectionParameters.m_ProjectionBias != nullptr)
    {
        constTensors.emplace_back(ConstTensor(managedProjectionBias.GetTensorInfo(),
                                              managedProjectionBias.Map()));
    }

    // Add norm parameters
    if (m_LayerNormParameters.m_InputLayerNormWeights != nullptr)
    {
        constTensors.emplace_back(ConstTensor(managedInputLayerNormWeights.GetTensorInfo(),
                                              managedInputLayerNormWeights.Map()));
    }
    if (m_LayerNormParameters.m_ForgetLayerNormWeights != nullptr)
    {
        constTensors.emplace_back(ConstTensor(managedForgetLayerNormWeights.GetTensorInfo(),
                                              managedForgetLayerNormWeights.Map()));
    }
    if (m_LayerNormParameters.m_CellLayerNormWeights != nullptr)
    {
        constTensors.emplace_back(ConstTensor(managedCellLayerNormWeights.GetTensorInfo(),
                                              managedCellLayerNormWeights.Map()));
    }
    if (m_LayerNormParameters.m_OutputLayerNormWeights != nullptr)
    {
        constTensors.emplace_back(ConstTensor(managedOutputLayerNormWeights.GetTensorInfo(),
                                              managedOutputLayerNormWeights.Map()));
    }
    strategy.ExecuteStrategy(this, GetParameters(), constTensors, GetName());
}

} // namespace armnn
