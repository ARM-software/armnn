//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//
#include "SoftmaxLayer.hpp"

#include "LayerCloneBase.hpp"

#include <armnn/TypesUtils.hpp>
#include <backendsCommon/WorkloadData.hpp>
#include <backendsCommon/WorkloadFactory.hpp>

namespace armnn
{

SoftmaxLayer::SoftmaxLayer(const SoftmaxDescriptor &param, const char* name)
    : LayerWithParameters(1, 1, LayerType::Softmax, param, name)
{
}

std::unique_ptr<IWorkload> SoftmaxLayer::CreateWorkload(const Graph& graph, const IWorkloadFactory& factory) const
{
    SoftmaxQueueDescriptor descriptor;
    return factory.CreateSoftmax(descriptor, PrepInfoAndDesc(descriptor, graph));
}

SoftmaxLayer* SoftmaxLayer::Clone(Graph& graph) const
{
    return CloneBase<SoftmaxLayer>(graph, m_Param, GetName());
}

void SoftmaxLayer::ValidateTensorShapesFromInputs()
{
    VerifyLayerConnections(1, CHECK_LOCATION());

    auto inferredShapes = InferOutputShapes({ GetInputSlot(0).GetConnection()->GetTensorInfo().GetShape() });

    BOOST_ASSERT(inferredShapes.size() == 1);

    ConditionalThrowIfNotEqual<LayerValidationException>(
        "SoftmaxLayer: TensorShape set on OutputSlot[0] does not match the inferred shape.",
        GetOutputSlot(0).GetTensorInfo().GetShape(),
        inferredShapes[0]);
}

void SoftmaxLayer::Accept(ILayerVisitor& visitor) const
{
    visitor.VisitSoftmaxLayer(this, GetParameters(), GetName());
}

} // namespace armnn
