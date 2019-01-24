//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//
#include "PermuteLayer.hpp"

#include "LayerCloneBase.hpp"

#include <armnn/TypesUtils.hpp>
#include <backendsCommon/WorkloadData.hpp>
#include <backendsCommon/WorkloadFactory.hpp>

#include <Permute.hpp>

namespace armnn
{

PermuteLayer::PermuteLayer(const PermuteDescriptor& param, const char* name)
    : LayerWithParameters(1, 1, LayerType::Permute, param, name)
{
}

std::unique_ptr<IWorkload> PermuteLayer::CreateWorkload(const Graph& graph, const IWorkloadFactory& factory) const
{
    PermuteQueueDescriptor descriptor;
    return factory.CreatePermute(descriptor, PrepInfoAndDesc(descriptor, graph));
}

PermuteLayer* PermuteLayer::Clone(Graph& graph) const
{
    return CloneBase<PermuteLayer>(graph, m_Param, GetName());
}

std::vector<TensorShape> PermuteLayer::InferOutputShapes(const std::vector<TensorShape>& inputShapes) const
{
    BOOST_ASSERT(inputShapes.size() == 1);
    const TensorShape& inShape = inputShapes[0];
    return std::vector<TensorShape> ({armnnUtils::Permuted(inShape, m_Param.m_DimMappings)});
}

void PermuteLayer::ValidateTensorShapesFromInputs()
{
    VerifyLayerConnections(1, CHECK_LOCATION());

    auto inferredShapes = InferOutputShapes({ GetInputSlot(0).GetConnection()->GetTensorInfo().GetShape() });

    BOOST_ASSERT(inferredShapes.size() == 1);

    ConditionalThrowIfNotEqual<LayerValidationException>(
        "PermuteLayer: TensorShape set on OutputSlot[0] does not match the inferred shape.",
        GetOutputSlot(0).GetTensorInfo().GetShape(),
        inferredShapes[0]);
}

void PermuteLayer::Accept(ILayerVisitor& visitor) const
{
    visitor.VisitPermuteLayer(this, GetParameters(), GetName());
}

} // namespace armnn
