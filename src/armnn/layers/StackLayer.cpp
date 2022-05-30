//
// Copyright © 2017 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//
#include "StackLayer.hpp"
#include "LayerCloneBase.hpp"

#include <armnn/TypesUtils.hpp>
#include <armnn/backends/WorkloadData.hpp>
#include <armnn/backends/WorkloadFactory.hpp>

#include <queue>

namespace armnn
{

StackLayer::StackLayer(const StackDescriptor& param, const char* name)
    : LayerWithParameters(param.m_NumInputs, 1, LayerType::Stack, param, name)
{
}

std::unique_ptr<IWorkload> StackLayer::CreateWorkload(const IWorkloadFactory& factory) const
{
    StackQueueDescriptor descriptor;
    SetAdditionalInfo(descriptor);

    return factory.CreateWorkload(LayerType::Stack, descriptor, PrepInfoAndDesc(descriptor));
}

StackLayer* StackLayer::Clone(Graph& graph) const
{
    return CloneBase<StackLayer>(graph, m_Param, GetName());
}

std::vector<TensorShape> StackLayer::InferOutputShapes(const std::vector<TensorShape>& inputShapes) const
{
    IgnoreUnused(inputShapes);

    const TensorShape& inputShape = m_Param.m_InputShape;
    const unsigned int inputNumDimensions = inputShape.GetNumDimensions();
    const unsigned int axis = m_Param.m_Axis;

    ARMNN_ASSERT(axis <= inputNumDimensions);

    std::vector<unsigned int> dimensionSizes(inputNumDimensions + 1, 0);
    for (unsigned int i = 0; i < axis; ++i)
    {
        dimensionSizes[i] = inputShape[i];
    }

    dimensionSizes[axis] = m_Param.m_NumInputs;

    for (unsigned int i = axis + 1; i < inputNumDimensions + 1; ++i)
    {
        dimensionSizes[i] = inputShape[i-1];
    }

    TensorShape targetShape = TensorShape(inputNumDimensions + 1, dimensionSizes.data());

    return std::vector<TensorShape>({ targetShape });
}

void StackLayer::ValidateTensorShapesFromInputs()
{
    // Validates Stack layer.
    ConditionalThrowIfNotEqual<LayerValidationException>(
        "StackLayer: Num Input Slots must match Num Inputs.",
        m_Param.m_NumInputs,
        GetNumInputSlots());

    VerifyLayerConnections(m_Param.m_NumInputs, CHECK_LOCATION());

    const TensorShape& outputShape = GetOutputSlot(0).GetTensorInfo().GetShape();

    VerifyShapeInferenceType(outputShape, m_ShapeInferenceMethod);

    // Constructs and validates input shapes
    std::vector<TensorShape> inputShapes;
    for (unsigned int i = 0; i < GetNumInputSlots(); ++i)
    {
        TensorShape inputShape = GetInputSlot(i).GetConnection()->GetTensorInfo().GetShape();
        if (inputShape != m_Param.m_InputShape)
        {
            throw LayerValidationException("StackLayer: TensorShape set on InputSlot[" +
                                           std::to_string(i) +
                                           "] does not match defined input shape");
        }
        inputShapes.push_back(inputShape);
    }

    auto inferredShapes = InferOutputShapes(inputShapes);

    ARMNN_ASSERT(inferredShapes.size() == 1);

    ValidateAndCopyShape(outputShape, inferredShapes[0], m_ShapeInferenceMethod, "StackLayer");
}

void StackLayer::ExecuteStrategy(IStrategy& strategy) const
{
    strategy.ExecuteStrategy(this, GetParameters(), {}, GetName());
}

} // namespace armnn armnn
