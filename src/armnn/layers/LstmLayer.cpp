//
// Copyright © 2017 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//
#include "LstmLayer.hpp"

#include "LayerCloneBase.hpp"

#include <armnn/LstmParams.hpp>
#include <armnn/TypesUtils.hpp>
#include <armnn/backends/TensorHandle.hpp>
#include <armnn/backends/WorkloadFactory.hpp>

namespace armnn
{

LstmLayer::LstmLayer(const LstmDescriptor& param, const char* name)
        : LayerWithParameters(3, 4, LayerType::Lstm, param, name)
{
}

std::unique_ptr<IWorkload> LstmLayer::CreateWorkload(const IWorkloadFactory& factory) const
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
        descriptor.m_CellLayerNormWeights = m_LayerNormParameters.m_CellLayerNormWeights.get();
        descriptor.m_OutputLayerNormWeights = m_LayerNormParameters.m_OutputLayerNormWeights.get();
    }

    SetAdditionalInfo(descriptor);

    return factory.CreateWorkload(LayerType::Lstm, descriptor, PrepInfoAndDesc(descriptor));
}

LstmLayer* LstmLayer::Clone(Graph& graph) const
{
    auto layer = CloneBase<LstmLayer>(graph, m_Param, GetName());

    layer->m_BasicParameters.m_InputToForgetWeights = m_BasicParameters.m_InputToForgetWeights ?
            m_BasicParameters.m_InputToForgetWeights
                : nullptr;
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
        if (!m_Param.m_CifgEnabled)
        {
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
        layer->m_LayerNormParameters.m_InputLayerNormWeights = m_LayerNormParameters.m_InputLayerNormWeights ?
               m_LayerNormParameters.m_InputLayerNormWeights : nullptr;
        layer->m_LayerNormParameters.m_ForgetLayerNormWeights = m_LayerNormParameters.m_ForgetLayerNormWeights ?
               m_LayerNormParameters.m_ForgetLayerNormWeights : nullptr;
        layer->m_LayerNormParameters.m_CellLayerNormWeights = m_LayerNormParameters.m_CellLayerNormWeights ?
               m_LayerNormParameters.m_CellLayerNormWeights : nullptr;
        layer->m_LayerNormParameters.m_OutputLayerNormWeights = m_LayerNormParameters.m_OutputLayerNormWeights ?
               m_LayerNormParameters.m_OutputLayerNormWeights : nullptr;
    }

    return std::move(layer);
}

std::vector<TensorShape> LstmLayer::InferOutputShapes(const std::vector<TensorShape>& inputShapes) const
{
    ARMNN_ASSERT(inputShapes.size() == 3);

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

    const TensorShape& outputShape = GetOutputSlot(0).GetTensorInfo().GetShape();

    VerifyShapeInferenceType(outputShape, m_ShapeInferenceMethod);

    auto inferredShapes = InferOutputShapes( {
        GetInputSlot(0).GetConnection()->GetTensorInfo().GetShape(),
        GetInputSlot(1).GetConnection()->GetTensorInfo().GetShape(),
        GetInputSlot(2).GetConnection()->GetTensorInfo().GetShape()
    });

    ARMNN_ASSERT(inferredShapes.size() == 4);

    // Check if the weights are nullptr
    ARMNN_ASSERT_MSG(m_BasicParameters.m_InputToForgetWeights != nullptr,
                     "LstmLayer: m_BasicParameters.m_InputToForgetWeights should not be null.");
    ARMNN_ASSERT_MSG(m_BasicParameters.m_InputToCellWeights != nullptr,
                     "LstmLayer: m_BasicParameters.m_InputToCellWeights should not be null.");
    ARMNN_ASSERT_MSG(m_BasicParameters.m_InputToOutputWeights != nullptr,
                     "LstmLayer: m_BasicParameters.m_InputToOutputWeights should not be null.");
    ARMNN_ASSERT_MSG(m_BasicParameters.m_RecurrentToForgetWeights != nullptr,
                     "LstmLayer: m_BasicParameters.m_RecurrentToForgetWeights should not be null.");
    ARMNN_ASSERT_MSG(m_BasicParameters.m_RecurrentToCellWeights != nullptr,
                     "LstmLayer: m_BasicParameters.m_RecurrentToCellWeights should not be null.");
    ARMNN_ASSERT_MSG(m_BasicParameters.m_RecurrentToOutputWeights != nullptr,
                     "LstmLayer: m_BasicParameters.m_RecurrentToOutputWeights should not be null.");
    ARMNN_ASSERT_MSG(m_BasicParameters.m_ForgetGateBias != nullptr,
                     "LstmLayer: m_BasicParameters.m_ForgetGateBias should not be null.");
    ARMNN_ASSERT_MSG(m_BasicParameters.m_CellBias != nullptr,
                     "LstmLayer: m_BasicParameters.m_CellBias should not be null.");
    ARMNN_ASSERT_MSG(m_BasicParameters.m_OutputGateBias != nullptr,
                     "LstmLayer: m_BasicParameters.m_OutputGateBias should not be null.");

    if (!m_Param.m_CifgEnabled)
    {
        ARMNN_ASSERT_MSG(m_CifgParameters.m_InputToInputWeights != nullptr,
                         "LstmLayer: m_CifgParameters.m_InputToInputWeights should not be null.");
        ARMNN_ASSERT_MSG(m_CifgParameters.m_RecurrentToInputWeights != nullptr,
                         "LstmLayer: m_CifgParameters.m_RecurrentToInputWeights should not be null.");
        ARMNN_ASSERT_MSG(m_CifgParameters.m_InputGateBias != nullptr,
                         "LstmLayer: m_CifgParameters.m_InputGateBias should not be null.");

        ValidateAndCopyShape(outputShape, inferredShapes[0], m_ShapeInferenceMethod, "LstmLayer");
    }
    else
    {
        ARMNN_ASSERT_MSG(m_CifgParameters.m_InputToInputWeights == nullptr,
            "LstmLayer: m_CifgParameters.m_InputToInputWeights should not have a value when CIFG is enabled.");
        ARMNN_ASSERT_MSG(m_CifgParameters.m_RecurrentToInputWeights == nullptr,
            "LstmLayer: m_CifgParameters.m_RecurrentToInputWeights should not have a value when CIFG is enabled.");
        ARMNN_ASSERT_MSG(m_CifgParameters.m_InputGateBias == nullptr,
            "LstmLayer: m_CifgParameters.m_InputGateBias should not have a value when CIFG is enabled.");

        ValidateAndCopyShape(outputShape, inferredShapes[0], m_ShapeInferenceMethod, "LstmLayer");
    }

    if (m_Param.m_ProjectionEnabled)
    {
        ARMNN_ASSERT_MSG(m_ProjectionParameters.m_ProjectionWeights != nullptr,
                         "LstmLayer: m_ProjectionParameters.m_ProjectionWeights should not be null.");
    }

    if (m_Param.m_PeepholeEnabled)
    {
        if (!m_Param.m_CifgEnabled)
        {
            ARMNN_ASSERT_MSG(m_PeepholeParameters.m_CellToInputWeights != nullptr,
                             "LstmLayer: m_PeepholeParameters.m_CellToInputWeights should not be null "
                             "when Peephole is enabled and CIFG is disabled.");
        }
        ARMNN_ASSERT_MSG(m_PeepholeParameters.m_CellToForgetWeights != nullptr,
                         "LstmLayer: m_PeepholeParameters.m_CellToForgetWeights should not be null.");
        ARMNN_ASSERT_MSG(m_PeepholeParameters.m_CellToOutputWeights != nullptr,
                         "LstmLayer: m_PeepholeParameters.m_CellToOutputWeights should not be null.");
    }

    ValidateAndCopyShape(
            GetOutputSlot(1).GetTensorInfo().GetShape(), inferredShapes[1], m_ShapeInferenceMethod, "LstmLayer", 1);
    ValidateAndCopyShape(
            GetOutputSlot(2).GetTensorInfo().GetShape(), inferredShapes[2], m_ShapeInferenceMethod, "LstmLayer", 2);
    ValidateAndCopyShape(
            GetOutputSlot(3).GetTensorInfo().GetShape(), inferredShapes[3], m_ShapeInferenceMethod, "LstmLayer", 3);

    if (m_Param.m_LayerNormEnabled)
    {
        if(!m_Param.m_CifgEnabled)
        {
            ARMNN_ASSERT_MSG(m_LayerNormParameters.m_InputLayerNormWeights != nullptr,
                             "LstmLayer: m_LayerNormParameters.m_inputLayerNormWeights should not be null.");
        }
        ARMNN_ASSERT_MSG(m_LayerNormParameters.m_ForgetLayerNormWeights != nullptr,
                         "LstmLayer: m_LayerNormParameters.m_forgetLayerNormWeights should not be null.");
        ARMNN_ASSERT_MSG(m_LayerNormParameters.m_CellLayerNormWeights != nullptr,
                         "LstmLayer: m_LayerNormParameters.m_cellLayerNormWeights should not be null.");
        ARMNN_ASSERT_MSG(m_LayerNormParameters.m_OutputLayerNormWeights != nullptr,
                         "LstmLayer: m_LayerNormParameters.m_outputLayerNormWeights should not be null.");
    }
}

Layer::ConstantTensors LstmLayer::GetConstantTensorsByRef()
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

ARMNN_NO_DEPRECATE_WARN_BEGIN
void LstmLayer::Accept(ILayerVisitor& visitor) const
{
    LstmInputParams inputParams;
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

    ConstTensor inputToInputWeightsTensor;
    if (m_CifgParameters.m_InputToInputWeights != nullptr)
    {
        ConstTensor inputToInputWeightsTensorCopy(managedInputToInputWeights.GetTensorInfo(),
                                                  managedInputToInputWeights.Map());
        inputToInputWeightsTensor = inputToInputWeightsTensorCopy;
        inputParams.m_InputToInputWeights = &inputToInputWeightsTensor;
    }
    ConstTensor inputToForgetWeightsTensor;
    if (m_BasicParameters.m_InputToForgetWeights != nullptr)
    {
        ConstTensor inputToForgetWeightsTensorCopy(managedInputToForgetWeights.GetTensorInfo(),
                                                   managedInputToForgetWeights.Map());
        inputToForgetWeightsTensor = inputToForgetWeightsTensorCopy;
        inputParams.m_InputToForgetWeights = &inputToForgetWeightsTensor;
    }
    ConstTensor inputToCellWeightsTensor;
    if (m_BasicParameters.m_InputToCellWeights != nullptr)
    {
        ConstTensor inputToCellWeightsTensorCopy(managedInputToCellWeights.GetTensorInfo(),
                                                 managedInputToCellWeights.Map());
        inputToCellWeightsTensor = inputToCellWeightsTensorCopy;
        inputParams.m_InputToCellWeights = &inputToCellWeightsTensor;
    }
    ConstTensor inputToOutputWeightsTensor;
    if (m_BasicParameters.m_InputToOutputWeights != nullptr)
    {
        ConstTensor inputToOutputWeightsTensorCopy(managedInputToOutputWeights.GetTensorInfo(),
                                                   managedInputToOutputWeights.Map());
        inputToOutputWeightsTensor = inputToOutputWeightsTensorCopy;
        inputParams.m_InputToOutputWeights = &inputToOutputWeightsTensor;
    }
    ConstTensor recurrentToInputWeightsTensor;
    if (m_CifgParameters.m_RecurrentToInputWeights != nullptr)
    {
        ConstTensor recurrentToInputWeightsTensorCopy(
                managedRecurrentToInputWeights.GetTensorInfo(),
                managedRecurrentToInputWeights.Map());
        recurrentToInputWeightsTensor = recurrentToInputWeightsTensorCopy;
        inputParams.m_RecurrentToInputWeights = &recurrentToInputWeightsTensor;
    }
    ConstTensor recurrentToForgetWeightsTensor;
    if (m_BasicParameters.m_RecurrentToForgetWeights != nullptr)
    {
        ConstTensor recurrentToForgetWeightsTensorCopy(
                managedRecurrentToForgetWeights.GetTensorInfo(),
                managedRecurrentToForgetWeights.Map());
        recurrentToForgetWeightsTensor = recurrentToForgetWeightsTensorCopy;
        inputParams.m_RecurrentToForgetWeights = &recurrentToForgetWeightsTensor;
    }
    ConstTensor recurrentToCellWeightsTensor;
    if (m_BasicParameters.m_RecurrentToCellWeights != nullptr)
    {
        ConstTensor recurrentToCellWeightsTensorCopy(
                managedRecurrentToCellWeights.GetTensorInfo(),
                managedRecurrentToCellWeights.Map());
        recurrentToCellWeightsTensor = recurrentToCellWeightsTensorCopy;
        inputParams.m_RecurrentToCellWeights = &recurrentToCellWeightsTensor;
    }
    ConstTensor recurrentToOutputWeightsTensor;
    if (m_BasicParameters.m_RecurrentToOutputWeights != nullptr)
    {
        ConstTensor recurrentToOutputWeightsTensorCopy(
                managedRecurrentToOutputWeights.GetTensorInfo(),
                managedRecurrentToOutputWeights.Map());
        recurrentToOutputWeightsTensor = recurrentToOutputWeightsTensorCopy;
        inputParams.m_RecurrentToOutputWeights = &recurrentToOutputWeightsTensor;
    }
    ConstTensor cellToInputWeightsTensor;
    if (m_PeepholeParameters.m_CellToInputWeights != nullptr)
    {
        ConstTensor cellToInputWeightsTensorCopy(managedCellToInputWeights.GetTensorInfo(),
                                                 managedCellToInputWeights.Map());
        cellToInputWeightsTensor = cellToInputWeightsTensorCopy;
        inputParams.m_CellToInputWeights = &cellToInputWeightsTensor;
    }
    ConstTensor cellToForgetWeightsTensor;
    if (m_PeepholeParameters.m_CellToForgetWeights != nullptr)
    {
        ConstTensor cellToForgetWeightsTensorCopy(managedCellToForgetWeights.GetTensorInfo(),
                                                  managedCellToForgetWeights.Map());
        cellToForgetWeightsTensor = cellToForgetWeightsTensorCopy;
        inputParams.m_CellToForgetWeights = &cellToForgetWeightsTensor;
    }
    ConstTensor cellToOutputWeightsTensor;
    if (m_PeepholeParameters.m_CellToOutputWeights != nullptr)
    {
        ConstTensor cellToOutputWeightsTensorCopy(managedCellToOutputWeights.GetTensorInfo(),
                                                  managedCellToOutputWeights.Map());
        cellToOutputWeightsTensor = cellToOutputWeightsTensorCopy;
        inputParams.m_CellToOutputWeights = &cellToOutputWeightsTensor;
    }
    ConstTensor inputGateBiasTensor;
    if (m_CifgParameters.m_InputGateBias != nullptr)
    {
        ConstTensor inputGateBiasTensorCopy(managedInputGateBias.GetTensorInfo(),
                                        managedInputGateBias.Map());
        inputGateBiasTensor = inputGateBiasTensorCopy;
        inputParams.m_InputGateBias = &inputGateBiasTensor;
    }
    ConstTensor forgetGateBiasTensor;
    if (m_BasicParameters.m_ForgetGateBias != nullptr)
    {
        ConstTensor forgetGateBiasTensorCopy(managedForgetGateBias.GetTensorInfo(),
                                             managedForgetGateBias.Map());
        forgetGateBiasTensor = forgetGateBiasTensorCopy;
        inputParams.m_ForgetGateBias = &forgetGateBiasTensor;
    }
    ConstTensor cellBiasTensor;
    if (m_BasicParameters.m_CellBias != nullptr)
    {
        ConstTensor cellBiasTensorCopy(managedCellBias.GetTensorInfo(),
                                       managedCellBias.Map());
        cellBiasTensor = cellBiasTensorCopy;
        inputParams.m_CellBias = &cellBiasTensor;
    }
    ConstTensor outputGateBias;
    if (m_BasicParameters.m_OutputGateBias != nullptr)
    {
        ConstTensor outputGateBiasCopy(managedOutputGateBias.GetTensorInfo(),
                                       managedOutputGateBias.Map());
        outputGateBias = outputGateBiasCopy;
        inputParams.m_OutputGateBias = &outputGateBias;
    }
    ConstTensor projectionWeightsTensor;
    if (m_ProjectionParameters.m_ProjectionWeights != nullptr)
    {
        ConstTensor projectionWeightsTensorCopy(managedProjectionWeights.GetTensorInfo(),
                                                managedProjectionWeights.Map());
        projectionWeightsTensor = projectionWeightsTensorCopy;
        inputParams.m_ProjectionWeights = &projectionWeightsTensor;
    }
    ConstTensor projectionBiasTensor;
    if (m_ProjectionParameters.m_ProjectionBias != nullptr)
    {
        ConstTensor projectionBiasTensorCopy(managedProjectionBias.GetTensorInfo(),
                                             managedProjectionBias.Map());
        projectionBiasTensor = projectionBiasTensorCopy;
        inputParams.m_ProjectionBias = &projectionBiasTensor;
    }
    ConstTensor inputLayerNormTensor;
    if (m_LayerNormParameters.m_InputLayerNormWeights != nullptr)
    {
        ConstTensor inputLayerNormTensorCopy(managedInputLayerNormWeights.GetTensorInfo(),
                                             managedInputLayerNormWeights.Map());
        inputLayerNormTensor = inputLayerNormTensorCopy;
        inputParams.m_InputLayerNormWeights = &inputLayerNormTensor;
    }
    ConstTensor forgetLayerNormTensor;
    if (m_LayerNormParameters.m_ForgetLayerNormWeights != nullptr)
    {
        ConstTensor forgetLayerNormTensorCopy(managedForgetLayerNormWeights.GetTensorInfo(),
                                             managedForgetLayerNormWeights.Map());
        forgetLayerNormTensor = forgetLayerNormTensorCopy;
        inputParams.m_ForgetLayerNormWeights = &forgetLayerNormTensor;
    }
    ConstTensor cellLayerNormTensor;
    if (m_LayerNormParameters.m_CellLayerNormWeights != nullptr)
    {
        ConstTensor cellLayerNormTensorCopy(managedCellLayerNormWeights.GetTensorInfo(),
                                              managedCellLayerNormWeights.Map());
        cellLayerNormTensor = cellLayerNormTensorCopy;
        inputParams.m_CellLayerNormWeights = &cellLayerNormTensor;
    }
    ConstTensor outputLayerNormTensor;
    if (m_LayerNormParameters.m_OutputLayerNormWeights != nullptr)
    {
        ConstTensor outputLayerNormTensorCopy(managedOutputLayerNormWeights.GetTensorInfo(),
                                            managedOutputLayerNormWeights.Map());
        outputLayerNormTensor = outputLayerNormTensorCopy;
        inputParams.m_OutputLayerNormWeights = &outputLayerNormTensor;
    }


    visitor.VisitLstmLayer(this, GetParameters(), inputParams, GetName());
}
ARMNN_NO_DEPRECATE_WARN_END

void LstmLayer::ExecuteStrategy(IStrategy& strategy) const
{
    std::vector<ConstTensor> constTensors;

    LstmDescriptor descriptor = GetParameters();

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

    // Add cifg parameters
    if (!descriptor.m_CifgEnabled)
    {
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
    }

    // Add peephole parameters
    if (descriptor.m_PeepholeEnabled)
    {
        if (!descriptor.m_CifgEnabled)
        {
            if (m_PeepholeParameters.m_CellToInputWeights != nullptr)
            {
                constTensors.emplace_back(ConstTensor(managedCellToInputWeights.GetTensorInfo(),
                                                      managedCellToInputWeights.Map()));
            }
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
    }

    // Add projection parameters
    if (descriptor.m_ProjectionEnabled)
    {
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
    }

    // Add norm parameters
    if (descriptor.m_LayerNormEnabled)
    {
        if (!descriptor.m_CifgEnabled)
        {
            if (m_LayerNormParameters.m_InputLayerNormWeights != nullptr)
            {
                constTensors.emplace_back(ConstTensor(managedInputLayerNormWeights.GetTensorInfo(),
                                                      managedInputLayerNormWeights.Map()));
            }
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
    }

    strategy.ExecuteStrategy(this, GetParameters(), constTensors, GetName());
}

} // namespace armnn
