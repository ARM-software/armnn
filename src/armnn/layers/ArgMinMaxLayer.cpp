//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//
#include "ArgMinMaxLayer.hpp"

#include "LayerCloneBase.hpp"

#include <armnn/TypesUtils.hpp>
#include <backendsCommon/WorkloadData.hpp>
#include <backendsCommon/WorkloadFactory.hpp>

namespace armnn
{

ArgMinMaxLayer::ArgMinMaxLayer(const ArgMinMaxDescriptor& param, const char* name)
    : LayerWithParameters(1, 1, LayerType::ArgMinMax, param, name)
{
}

std::unique_ptr<IWorkload> ArgMinMaxLayer::CreateWorkload(const Graph& graph,
    const IWorkloadFactory& factory) const
{
    ArgMinMaxQueueDescriptor descriptor;
    return factory.CreateArgMinMax(descriptor, PrepInfoAndDesc(descriptor, graph));
}

ArgMinMaxLayer* ArgMinMaxLayer::Clone(Graph& graph) const
{
    return CloneBase<ArgMinMaxLayer>(graph, m_Param, GetName());
}

void ArgMinMaxLayer::ValidateTensorShapesFromInputs()
{
    VerifyLayerConnections(1, CHECK_LOCATION());

    auto inferredShapes = InferOutputShapes({ GetInputSlot(0).GetConnection()->GetTensorInfo().GetShape() });

    BOOST_ASSERT(inferredShapes.size() == 1);

    ConditionalThrowIfNotEqual<LayerValidationException>(
            "ArgMinMaxLayer: TensorShape set on OutputSlot does not match the inferred shape.",
            GetOutputSlot(0).GetTensorInfo().GetShape(),
            inferredShapes[0]);
}

void ArgMinMaxLayer::Accept(ILayerVisitor& visitor) const
{
    visitor.VisitArgMinMaxLayer(this, GetParameters(), GetName());
}

} // namespace armnn