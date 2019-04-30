//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//
#include "SwitchLayer.hpp"

#include "LayerCloneBase.hpp"

#include <backendsCommon/WorkloadData.hpp>
#include <backendsCommon/WorkloadFactory.hpp>

namespace armnn
{

SwitchLayer::SwitchLayer(const char* name)
    : Layer(2, 2, LayerType::Switch, name)
{}

std::unique_ptr<IWorkload> SwitchLayer::CreateWorkload(const Graph& graph,
                                                       const IWorkloadFactory& factory) const
{
    SwitchQueueDescriptor descriptor;
    return factory.CreateSwitch(descriptor, PrepInfoAndDesc(descriptor, graph));
}

SwitchLayer* SwitchLayer::Clone(Graph& graph) const
{
    return CloneBase<SwitchLayer>(graph, GetName());
}

void SwitchLayer::ValidateTensorShapesFromInputs()
{
    VerifyLayerConnections(2, CHECK_LOCATION());

    BOOST_ASSERT_MSG(GetNumOutputSlots() == 2, "SwitchLayer: The layer should return 2 outputs.");

    // Assuming first input is the Input and second input is the Constant
    std::vector<TensorShape> inferredShapes = InferOutputShapes({
        GetInputSlot(0).GetConnection()->GetTensorInfo().GetShape(),
        GetInputSlot(1).GetConnection()->GetTensorInfo().GetShape() });

    BOOST_ASSERT(inferredShapes.size() == 2);

    ConditionalThrowIfNotEqual<LayerValidationException>(
        "SwitchLayer: TensorShape set on OutputSlot[0] does not match the inferred shape.",
        GetOutputSlot(0).GetTensorInfo().GetShape(),
        inferredShapes[0]);

    ConditionalThrowIfNotEqual<LayerValidationException>(
        "SwitchLayer: TensorShape set on OutputSlot[0] does not match the inferred shape.",
        GetOutputSlot(1).GetTensorInfo().GetShape(),
        inferredShapes[0]);
}

void SwitchLayer::Accept(ILayerVisitor& visitor) const
{
    visitor.VisitSwitchLayer(this, GetName());
}

} // namespace armnn
