//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//
#include "FloorLayer.hpp"

#include "LayerCloneBase.hpp"

#include <armnn/TypesUtils.hpp>
#include <backendsCommon/WorkloadData.hpp>
#include <backendsCommon/WorkloadFactory.hpp>

namespace armnn
{

FloorLayer::FloorLayer(const char* name)
 : Layer(1, 1, LayerType::Floor, name)
{
}

std::unique_ptr<IWorkload> FloorLayer::CreateWorkload(const Graph& graph,
    const IWorkloadFactory& factory) const
{
    FloorQueueDescriptor descriptor;
    return factory.CreateFloor(descriptor, PrepInfoAndDesc(descriptor, graph));
}

FloorLayer* FloorLayer::Clone(Graph& graph) const
{
    return CloneBase<FloorLayer>(graph, GetName());
}

void FloorLayer::ValidateTensorShapesFromInputs()
{
    VerifyLayerConnections(1, CHECK_LOCATION());

    auto inferredShapes = InferOutputShapes({ GetInputSlot(0).GetConnection()->GetTensorInfo().GetShape() });

    BOOST_ASSERT(inferredShapes.size() == 1);

    ConditionalThrowIfNotEqual<LayerValidationException>(
        "FloorLayer: TensorShape set on OutputSlot[0] does not match the inferred shape.",
        GetOutputSlot(0).GetTensorInfo().GetShape(),
        inferredShapes[0]);
}

void FloorLayer::Accept(ILayerVisitor& visitor) const
{
    visitor.VisitFloorLayer(this, GetName());
}

} // namespace armnn
