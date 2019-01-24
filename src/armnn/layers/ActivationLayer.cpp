//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//
#include "ActivationLayer.hpp"
#include "LayerCloneBase.hpp"

#include <armnn/TypesUtils.hpp>
#include <backendsCommon/WorkloadData.hpp>
#include <backendsCommon/WorkloadFactory.hpp>

namespace armnn
{

ActivationLayer::ActivationLayer(const ActivationDescriptor& param, const char* name)
    : LayerWithParameters(1, 1, LayerType::Activation, param, name)
{
}

std::unique_ptr<IWorkload> ActivationLayer::CreateWorkload(const Graph& graph, const IWorkloadFactory& factory) const
{
    ActivationQueueDescriptor descriptor;
    return factory.CreateActivation(descriptor, PrepInfoAndDesc(descriptor, graph));
}

ActivationLayer* ActivationLayer::Clone(Graph& graph) const
{
    return CloneBase<ActivationLayer>(graph, m_Param, GetName());
}

void ActivationLayer::ValidateTensorShapesFromInputs()
{
    VerifyLayerConnections(1, CHECK_LOCATION());

    auto inferredShapes = InferOutputShapes({ GetInputSlot(0).GetConnection()->GetTensorInfo().GetShape() });

    BOOST_ASSERT(inferredShapes.size() == 1);

    ConditionalThrowIfNotEqual<LayerValidationException>(
        "ActivationLayer: TensorShape set on OutputSlot[0] does not match the inferred shape.",
        GetOutputSlot(0).GetTensorInfo().GetShape(),
        inferredShapes[0]);
}

void ActivationLayer::Accept(ILayerVisitor& visitor) const
{
    visitor.VisitActivationLayer(this, GetParameters(), GetName());
}

} // namespace armnn
