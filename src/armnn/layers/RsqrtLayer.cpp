//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "RsqrtLayer.hpp"

#include "LayerCloneBase.hpp"

#include <armnn/TypesUtils.hpp>
#include <backendsCommon/WorkloadData.hpp>
#include <backendsCommon/WorkloadFactory.hpp>

namespace armnn
{

RsqrtLayer::RsqrtLayer(const char* name)
    : Layer(1, 1, LayerType::Rsqrt, name)
{
}

std::unique_ptr<IWorkload> RsqrtLayer::CreateWorkload(const Graph& graph,
                                                      const IWorkloadFactory& factory) const
{
    RsqrtQueueDescriptor descriptor;
    return factory.CreateRsqrt(descriptor, PrepInfoAndDesc(descriptor, graph));
}

RsqrtLayer* RsqrtLayer::Clone(Graph& graph) const
{
    return CloneBase<RsqrtLayer>(graph, GetName());
}

void RsqrtLayer::ValidateTensorShapesFromInputs()
{
    VerifyLayerConnections(1, CHECK_LOCATION());

    auto inferredShapes = InferOutputShapes({ GetInputSlot(0).GetConnection()->GetTensorInfo().GetShape() });

    BOOST_ASSERT(inferredShapes.size() == 1);

    ConditionalThrowIfNotEqual<LayerValidationException>(
            "RsqrtLayer: TensorShape set on OutputSlot[0] does not match the inferred shape.",
            GetOutputSlot(0).GetTensorInfo().GetShape(),
            inferredShapes[0]);
}

void RsqrtLayer::Accept(ILayerVisitor& visitor) const
{
    visitor.VisitRsqrtLayer(this, GetName());
}

} // namespace armnn