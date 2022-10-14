//
// Copyright © 2017 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//
#include "DebugLayer.hpp"

#include "LayerCloneBase.hpp"

#include <armnn/backends/WorkloadData.hpp>
#include <armnn/backends/WorkloadFactory.hpp>

namespace armnn
{

DebugLayer::DebugLayer(const char* name)
    : Layer(1, 1, LayerType::Debug, name),
      m_ToFile(false)
{}

DebugLayer::DebugLayer(const char* name, bool toFile)
    : Layer(1, 1, LayerType::Debug, name),
      m_ToFile(toFile)
{}

std::unique_ptr<IWorkload> DebugLayer::CreateWorkload(const IWorkloadFactory& factory) const
{
    const Layer& prevLayer = GetInputSlot(0).GetConnectedOutputSlot()->GetOwningLayer();

    DebugQueueDescriptor descriptor;
    descriptor.m_Guid = prevLayer.GetGuid();
    descriptor.m_LayerName = prevLayer.GetNameStr();
    descriptor.m_SlotIndex = GetInputSlot(0).GetConnectedOutputSlot()->CalculateIndexOnOwner();
    descriptor.m_LayerOutputToFile = m_ToFile;

    SetAdditionalInfo(descriptor);

    return factory.CreateWorkload(LayerType::Debug, descriptor, PrepInfoAndDesc(descriptor));
}

DebugLayer* DebugLayer::Clone(Graph& graph) const
{
    return CloneBase<DebugLayer>(graph, GetName());
}

void DebugLayer::ValidateTensorShapesFromInputs()
{
    VerifyLayerConnections(1, CHECK_LOCATION());

    const TensorShape& outputShape = GetOutputSlot(0).GetTensorInfo().GetShape();

    VerifyShapeInferenceType(outputShape, m_ShapeInferenceMethod);

    std::vector<TensorShape> inferredShapes = InferOutputShapes({
        GetInputSlot(0).GetConnection()->GetTensorInfo().GetShape() });

    ARMNN_ASSERT(inferredShapes.size() == 1);

    ValidateAndCopyShape(outputShape, inferredShapes[0], m_ShapeInferenceMethod, "DebugLayer");
}

void DebugLayer::ExecuteStrategy(IStrategy& strategy) const
{
    strategy.ExecuteStrategy(this, GetParameters(), {}, GetName());
}

} // namespace armnn
