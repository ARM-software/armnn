//
// Copyright © 2017 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//
#include "DebugLayer.hpp"

#include "LayerCloneBase.hpp"

#include <armnn/backends/WorkloadData.hpp>
#include <armnn/backends/WorkloadFactory.hpp>
#include <armnn/utility/IgnoreUnused.hpp>

namespace armnn
{

DebugLayer::DebugLayer(const char* name)
    : Layer(1, 1, LayerType::Debug, name)
{}

std::unique_ptr<IWorkload> DebugLayer::CreateWorkload(const IWorkloadFactory& factory) const
{
    const Layer& prevLayer = GetInputSlot(0).GetConnectedOutputSlot()->GetOwningLayer();

    DebugQueueDescriptor descriptor;
    descriptor.m_Guid = prevLayer.GetGuid();
    descriptor.m_LayerName = prevLayer.GetNameStr();
    descriptor.m_SlotIndex = GetInputSlot(0).GetConnectedOutputSlot()->CalculateIndexOnOwner();

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

ARMNN_NO_DEPRECATE_WARN_BEGIN
void DebugLayer::Accept(ILayerVisitor& visitor) const
{
    // by design debug layers are never in input graphs
    IgnoreUnused(visitor);
    throw armnn::Exception("DebugLayer should never appear in an input graph");
}
ARMNN_NO_DEPRECATE_WARN_END

} // namespace armnn
