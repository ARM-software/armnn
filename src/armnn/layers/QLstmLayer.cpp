//
// Copyright © 2020 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//
#include "QLstmLayer.hpp"

#include "LayerCloneBase.hpp"

#include <armnn/LstmParams.hpp>
#include <armnn/TypesUtils.hpp>
#include <backendsCommon/CpuTensorHandle.hpp>
#include <backendsCommon/WorkloadFactory.hpp>

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

    return factory.CreateQLstm(descriptor, PrepInfoAndDesc(descriptor));
}

QLstmLayer* QLstmLayer::Clone(Graph& graph) const
{
    auto layer = CloneBase<QLstmLayer>(graph, m_Param, GetName());

    layer->m_BasicParameters.m_InputToForgetWeights = m_BasicParameters.m_InputToForgetWeights ?
            std::make_unique<ScopedCpuTensorHandle>(*m_BasicParameters.m_InputToForgetWeights) : nullptr;
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
        if (!m_Param.m_CifgEnabled) {
            layer->m_PeepholeParameters.m_CellToInputWeights = m_PeepholeParameters.m_CellToInputWeights ?
                    std::make_unique<ScopedCpuTensorHandle>(*m_PeepholeParameters.m_CellToInputWeights) : nullptr;
        }

        layer->m_PeepholeParameters.m_CellToForgetWeights = m_PeepholeParameters.m_CellToForgetWeights ?
                std::make_unique<ScopedCpuTensorHandle>(*m_PeepholeParameters.m_CellToForgetWeights) : nullptr;
        layer->m_PeepholeParameters.m_CellToOutputWeights = m_PeepholeParameters.m_CellToOutputWeights ?
                std::make_unique<ScopedCpuTensorHandle>(*m_PeepholeParameters.m_CellToOutputWeights) : nullptr;
    }

    if (m_Param.m_LayerNormEnabled)
    {
        if (!m_Param.m_CifgEnabled) {
            layer->m_LayerNormParameters.m_InputLayerNormWeights = m_LayerNormParameters.m_InputLayerNormWeights ?
                    std::make_unique<ScopedCpuTensorHandle>(*m_LayerNormParameters.m_InputLayerNormWeights) : nullptr;
        }

        layer->m_LayerNormParameters.m_ForgetLayerNormWeights = m_LayerNormParameters.m_ForgetLayerNormWeights ?
                std::make_unique<ScopedCpuTensorHandle>(*m_LayerNormParameters.m_ForgetLayerNormWeights) : nullptr;
        layer->m_LayerNormParameters.m_CellLayerNormWeights = m_LayerNormParameters.m_CellLayerNormWeights ?
                std::make_unique<ScopedCpuTensorHandle>(*m_LayerNormParameters.m_CellLayerNormWeights) : nullptr;
        layer->m_LayerNormParameters.m_OutputLayerNormWeights = m_LayerNormParameters.m_OutputLayerNormWeights ?
                std::make_unique<ScopedCpuTensorHandle>(*m_LayerNormParameters.m_OutputLayerNormWeights) : nullptr;
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

Layer::ConstantTensors QLstmLayer::GetConstantTensorsByRef()
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

void QLstmLayer::Accept(ILayerVisitor& visitor) const
{
    LstmInputParams inputParams;

    ConstTensor inputToInputWeightsTensor;
    if (m_CifgParameters.m_InputToInputWeights != nullptr)
    {
        ConstTensor inputToInputWeightsTensorCopy(m_CifgParameters.m_InputToInputWeights->GetTensorInfo(),
                                                  m_CifgParameters.m_InputToInputWeights->Map(true));
        inputToInputWeightsTensor = inputToInputWeightsTensorCopy;
        inputParams.m_InputToInputWeights = &inputToInputWeightsTensor;
    }

    ConstTensor inputToForgetWeightsTensor;
    if (m_BasicParameters.m_InputToForgetWeights != nullptr)
    {
        ConstTensor inputToForgetWeightsTensorCopy(m_BasicParameters.m_InputToForgetWeights->GetTensorInfo(),
                                                   m_BasicParameters.m_InputToForgetWeights->Map(true));
        inputToForgetWeightsTensor = inputToForgetWeightsTensorCopy;
        inputParams.m_InputToForgetWeights = &inputToForgetWeightsTensor;
    }

    ConstTensor inputToCellWeightsTensor;
    if (m_BasicParameters.m_InputToCellWeights != nullptr)
    {
        ConstTensor inputToCellWeightsTensorCopy(m_BasicParameters.m_InputToCellWeights->GetTensorInfo(),
                                                 m_BasicParameters.m_InputToCellWeights->Map(true));
        inputToCellWeightsTensor = inputToCellWeightsTensorCopy;
        inputParams.m_InputToCellWeights = &inputToCellWeightsTensor;
    }

    ConstTensor inputToOutputWeightsTensor;
    if (m_BasicParameters.m_InputToOutputWeights != nullptr)
    {
        ConstTensor inputToOutputWeightsTensorCopy(m_BasicParameters.m_InputToOutputWeights->GetTensorInfo(),
                                                   m_BasicParameters.m_InputToOutputWeights->Map(true));
        inputToOutputWeightsTensor = inputToOutputWeightsTensorCopy;
        inputParams.m_InputToOutputWeights = &inputToOutputWeightsTensor;
    }

    ConstTensor recurrentToInputWeightsTensor;
    if (m_CifgParameters.m_RecurrentToInputWeights != nullptr)
    {
        ConstTensor recurrentToInputWeightsTensorCopy(
                m_CifgParameters.m_RecurrentToInputWeights->GetTensorInfo(),
                m_CifgParameters.m_RecurrentToInputWeights->Map(true));
        recurrentToInputWeightsTensor = recurrentToInputWeightsTensorCopy;
        inputParams.m_RecurrentToInputWeights = &recurrentToInputWeightsTensor;
    }

    ConstTensor recurrentToForgetWeightsTensor;
    if (m_BasicParameters.m_RecurrentToForgetWeights != nullptr)
    {
        ConstTensor recurrentToForgetWeightsTensorCopy(
                m_BasicParameters.m_RecurrentToForgetWeights->GetTensorInfo(),
                m_BasicParameters.m_RecurrentToForgetWeights->Map(true));
        recurrentToForgetWeightsTensor = recurrentToForgetWeightsTensorCopy;
        inputParams.m_RecurrentToForgetWeights = &recurrentToForgetWeightsTensor;
    }

    ConstTensor recurrentToCellWeightsTensor;
    if (m_BasicParameters.m_RecurrentToCellWeights != nullptr)
    {
        ConstTensor recurrentToCellWeightsTensorCopy(
                m_BasicParameters.m_RecurrentToCellWeights->GetTensorInfo(),
                m_BasicParameters.m_RecurrentToCellWeights->Map(true));
        recurrentToCellWeightsTensor = recurrentToCellWeightsTensorCopy;
        inputParams.m_RecurrentToCellWeights = &recurrentToCellWeightsTensor;
    }

    ConstTensor recurrentToOutputWeightsTensor;
    if (m_BasicParameters.m_RecurrentToOutputWeights != nullptr)
    {
        ConstTensor recurrentToOutputWeightsTensorCopy(
                m_BasicParameters.m_RecurrentToOutputWeights->GetTensorInfo(),
                m_BasicParameters.m_RecurrentToOutputWeights->Map(true));
        recurrentToOutputWeightsTensor = recurrentToOutputWeightsTensorCopy;
        inputParams.m_RecurrentToOutputWeights = &recurrentToOutputWeightsTensor;
    }

    ConstTensor cellToInputWeightsTensor;
    if (m_PeepholeParameters.m_CellToInputWeights != nullptr)
    {
        ConstTensor cellToInputWeightsTensorCopy(m_PeepholeParameters.m_CellToInputWeights->GetTensorInfo(),
                                                 m_PeepholeParameters.m_CellToInputWeights->Map(true));
        cellToInputWeightsTensor = cellToInputWeightsTensorCopy;
        inputParams.m_CellToInputWeights = &cellToInputWeightsTensor;
    }

    ConstTensor cellToForgetWeightsTensor;
    if (m_PeepholeParameters.m_CellToForgetWeights != nullptr)
    {
        ConstTensor cellToForgetWeightsTensorCopy(m_PeepholeParameters.m_CellToForgetWeights->GetTensorInfo(),
                                                  m_PeepholeParameters.m_CellToForgetWeights->Map(true));
        cellToForgetWeightsTensor = cellToForgetWeightsTensorCopy;
        inputParams.m_CellToForgetWeights = &cellToForgetWeightsTensor;
    }

    ConstTensor cellToOutputWeightsTensor;
    if (m_PeepholeParameters.m_CellToOutputWeights != nullptr)
    {
        ConstTensor cellToOutputWeightsTensorCopy(m_PeepholeParameters.m_CellToOutputWeights->GetTensorInfo(),
                                                  m_PeepholeParameters.m_CellToOutputWeights->Map(true));
        cellToOutputWeightsTensor = cellToOutputWeightsTensorCopy;
        inputParams.m_CellToOutputWeights = &cellToOutputWeightsTensor;
    }

    ConstTensor inputGateBiasTensor;
    if (m_CifgParameters.m_InputGateBias != nullptr)
    {
        ConstTensor inputGateBiasTensorCopy(m_CifgParameters.m_InputGateBias->GetTensorInfo(),
                                            m_CifgParameters.m_InputGateBias->Map(true));
        inputGateBiasTensor = inputGateBiasTensorCopy;
        inputParams.m_InputGateBias = &inputGateBiasTensor;
    }

    ConstTensor forgetGateBiasTensor;
    if (m_BasicParameters.m_ForgetGateBias != nullptr)
    {
        ConstTensor forgetGateBiasTensorCopy(m_BasicParameters.m_ForgetGateBias->GetTensorInfo(),
                                             m_BasicParameters.m_ForgetGateBias->Map(true));
        forgetGateBiasTensor = forgetGateBiasTensorCopy;
        inputParams.m_ForgetGateBias = &forgetGateBiasTensor;
    }

    ConstTensor cellBiasTensor;
    if (m_BasicParameters.m_CellBias != nullptr)
    {
        ConstTensor cellBiasTensorCopy(m_BasicParameters.m_CellBias->GetTensorInfo(),
                                       m_BasicParameters.m_CellBias->Map(true));
        cellBiasTensor = cellBiasTensorCopy;
        inputParams.m_CellBias = &cellBiasTensor;
    }

    ConstTensor outputGateBias;
    if (m_BasicParameters.m_OutputGateBias != nullptr)
    {
        ConstTensor outputGateBiasCopy(m_BasicParameters.m_OutputGateBias->GetTensorInfo(),
                                       m_BasicParameters.m_OutputGateBias->Map(true));
        outputGateBias = outputGateBiasCopy;
        inputParams.m_OutputGateBias = &outputGateBias;
    }

    ConstTensor projectionWeightsTensor;
    if (m_ProjectionParameters.m_ProjectionWeights != nullptr)
    {
        ConstTensor projectionWeightsTensorCopy(m_ProjectionParameters.m_ProjectionWeights->GetTensorInfo(),
                                                m_ProjectionParameters.m_ProjectionWeights->Map(true));
        projectionWeightsTensor = projectionWeightsTensorCopy;
        inputParams.m_ProjectionWeights = &projectionWeightsTensor;
    }

    ConstTensor projectionBiasTensor;
    if (m_ProjectionParameters.m_ProjectionBias != nullptr)
    {
        ConstTensor projectionBiasTensorCopy(m_ProjectionParameters.m_ProjectionBias->GetTensorInfo(),
                                             m_ProjectionParameters.m_ProjectionBias->Map(true));
        projectionBiasTensor = projectionBiasTensorCopy;
        inputParams.m_ProjectionBias = &projectionBiasTensor;
    }

    ConstTensor inputLayerNormTensor;
    if (m_LayerNormParameters.m_InputLayerNormWeights != nullptr)
    {
        ConstTensor inputLayerNormTensorCopy(m_LayerNormParameters.m_InputLayerNormWeights->GetTensorInfo(),
                                             m_LayerNormParameters.m_InputLayerNormWeights->Map(true));
        inputLayerNormTensor = inputLayerNormTensorCopy;
        inputParams.m_InputLayerNormWeights = &inputLayerNormTensor;
    }

    ConstTensor forgetLayerNormTensor;
    if (m_LayerNormParameters.m_ForgetLayerNormWeights != nullptr)
    {
        ConstTensor forgetLayerNormTensorCopy(m_LayerNormParameters.m_ForgetLayerNormWeights->GetTensorInfo(),
                                              m_LayerNormParameters.m_ForgetLayerNormWeights->Map(true));
        forgetLayerNormTensor = forgetLayerNormTensorCopy;
        inputParams.m_ForgetLayerNormWeights = &forgetLayerNormTensor;
    }

    ConstTensor cellLayerNormTensor;
    if (m_LayerNormParameters.m_CellLayerNormWeights != nullptr)
    {
        ConstTensor cellLayerNormTensorCopy(m_LayerNormParameters.m_CellLayerNormWeights->GetTensorInfo(),
                                            m_LayerNormParameters.m_CellLayerNormWeights->Map(true));
        cellLayerNormTensor = cellLayerNormTensorCopy;
        inputParams.m_CellLayerNormWeights = &cellLayerNormTensor;
    }

    ConstTensor outputLayerNormTensor;
    if (m_LayerNormParameters.m_OutputLayerNormWeights != nullptr)
    {
        ConstTensor outputLayerNormTensorCopy(m_LayerNormParameters.m_OutputLayerNormWeights->GetTensorInfo(),
                                              m_LayerNormParameters.m_OutputLayerNormWeights->Map(true));
        outputLayerNormTensor = outputLayerNormTensorCopy;
        inputParams.m_OutputLayerNormWeights = &outputLayerNormTensor;
    }


    visitor.VisitQLstmLayer(this, GetParameters(), inputParams, GetName());
}

} // namespace armnn
