//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//
#include "DebugLayer.hpp"

#include "LayerCloneBase.hpp"

#include <backendsCommon/WorkloadData.hpp>
#include <backendsCommon/WorkloadFactory.hpp>

namespace armnn
{

DebugLayer::DebugLayer(const char* name)
    : Layer(1, 1, LayerType::Debug, name)
{}

std::unique_ptr<IWorkload> DebugLayer::CreateWorkload(const Graph& graph,
                                                      const IWorkloadFactory& factory) const
{
    const Layer& prevLayer = GetInputSlot(0).GetConnectedOutputSlot()->GetOwningLayer();

    DebugQueueDescriptor descriptor;
    descriptor.m_Guid = prevLayer.GetGuid();
    descriptor.m_LayerName = prevLayer.GetNameStr();
    descriptor.m_SlotIndex = GetInputSlot(0).GetConnectedOutputSlot()->CalculateIndexOnOwner();

    return factory.CreateDebug(descriptor, PrepInfoAndDesc(descriptor, graph));
}

DebugLayer* DebugLayer::Clone(Graph& graph) const
{
    return CloneBase<DebugLayer>(graph, GetName());
}

void DebugLayer::ValidateTensorShapesFromInputs()
{
    VerifyLayerConnections(1, CHECK_LOCATION());

    std::vector<TensorShape> inferredShapes = InferOutputShapes({
        GetInputSlot(0).GetConnection()->GetTensorInfo().GetShape() });

    BOOST_ASSERT(inferredShapes.size() == 1);

    ConditionalThrowIfNotEqual<LayerValidationException>(
        "DebugLayer: TensorShape set on OutputSlot[0] does not match the inferred shape.",
        GetOutputSlot(0).GetTensorInfo().GetShape(),
        inferredShapes[0]);
}

void DebugLayer::Accept(ILayerVisitor& visitor) const
{
    // by design debug layers are never in input graphs
    throw armnn::Exception("DebugLayer should never appear in an input graph");
}

} // namespace armnn
