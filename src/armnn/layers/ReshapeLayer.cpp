//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//
#include "ReshapeLayer.hpp"

#include "LayerCloneBase.hpp"

#include <armnn/TypesUtils.hpp>
#include <backendsCommon/WorkloadData.hpp>
#include <backendsCommon/WorkloadFactory.hpp>

namespace armnn
{

ReshapeLayer::ReshapeLayer(const ReshapeDescriptor& param, const char* name)
    : LayerWithParameters(1, 1, LayerType::Reshape, param, name)
{
}

std::unique_ptr<IWorkload> ReshapeLayer::CreateWorkload(const Graph& graph,
    const IWorkloadFactory& factory) const
{
    ReshapeQueueDescriptor descriptor;
    return factory.CreateReshape(descriptor, PrepInfoAndDesc(descriptor, graph));
}

ReshapeLayer* ReshapeLayer::Clone(Graph& graph) const
{
    return CloneBase<ReshapeLayer>(graph, m_Param, GetName());
}

std::vector<TensorShape> ReshapeLayer::InferOutputShapes(const std::vector<TensorShape>& inputShapes) const
{
    return std::vector<TensorShape>({ m_Param.m_TargetShape });
}

void ReshapeLayer::ValidateTensorShapesFromInputs()
{
    VerifyLayerConnections(1, CHECK_LOCATION());

    auto inferredShapes = InferOutputShapes({  });

    BOOST_ASSERT(inferredShapes.size() == 1);

    ConditionalThrowIfNotEqual<LayerValidationException>(
        "ReshapeLayer: TensorShape set on OutputSlot[0] does not match the inferred shape.",
        GetOutputSlot(0).GetTensorInfo().GetShape(),
        inferredShapes[0]);
}

void ReshapeLayer::Accept(ILayerVisitor& visitor) const
{
    visitor.VisitReshapeLayer(this, GetParameters(), GetName());
}

} // namespace armnn
