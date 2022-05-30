//
// Copyright © 2017 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//
#include "SwitchLayer.hpp"

#include "LayerCloneBase.hpp"

#include <armnn/backends/WorkloadData.hpp>
#include <armnn/backends/WorkloadFactory.hpp>

namespace armnn
{

SwitchLayer::SwitchLayer(const char* name)
    : Layer(2, 2, LayerType::Switch, name)
{}

std::unique_ptr<IWorkload> SwitchLayer::CreateWorkload(const IWorkloadFactory& factory) const
{
    SwitchQueueDescriptor descriptor;
    SetAdditionalInfo(descriptor);

    return factory.CreateWorkload(LayerType::Switch, descriptor, PrepInfoAndDesc(descriptor));
}

SwitchLayer* SwitchLayer::Clone(Graph& graph) const
{
    return CloneBase<SwitchLayer>(graph, GetName());
}

void SwitchLayer::ValidateTensorShapesFromInputs()
{
    VerifyLayerConnections(2, CHECK_LOCATION());

    const TensorShape& outputShape = GetOutputSlot(0).GetTensorInfo().GetShape();

    VerifyShapeInferenceType(outputShape, m_ShapeInferenceMethod);

    ARMNN_ASSERT_MSG(GetNumOutputSlots() == 2, "SwitchLayer: The layer should return 2 outputs.");

    // Assuming first input is the Input and second input is the Constant
    std::vector<TensorShape> inferredShapes = InferOutputShapes({
        GetInputSlot(0).GetConnection()->GetTensorInfo().GetShape(),
        GetInputSlot(1).GetConnection()->GetTensorInfo().GetShape()});

    ARMNN_ASSERT(inferredShapes.size() == 2);

    ValidateAndCopyShape(outputShape, inferredShapes[0], m_ShapeInferenceMethod, "SwitchLayer");

    ValidateAndCopyShape(
            GetOutputSlot(1).GetTensorInfo().GetShape(), inferredShapes[1], m_ShapeInferenceMethod, "SwitchLayer", 1);
}

void SwitchLayer::ExecuteStrategy(IStrategy& strategy) const
{
    strategy.ExecuteStrategy(this, GetParameters(), {}, GetName());
}

} // namespace armnn
