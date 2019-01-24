//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "GreaterLayer.hpp"

#include "LayerCloneBase.hpp"

#include <armnn/TypesUtils.hpp>
#include <backendsCommon/WorkloadData.hpp>
#include <backendsCommon/WorkloadFactory.hpp>

namespace armnn
{

GreaterLayer::GreaterLayer(const char* name)
    : ElementwiseBaseLayer(2, 1, LayerType::Greater, name)
{
}

std::unique_ptr<IWorkload> GreaterLayer::CreateWorkload(const Graph& graph,
                                                        const IWorkloadFactory& factory) const
{
    GreaterQueueDescriptor descriptor;
    return factory.CreateGreater(descriptor, PrepInfoAndDesc(descriptor, graph));
}

GreaterLayer* GreaterLayer::Clone(Graph& graph) const
{
    return CloneBase<GreaterLayer>(graph, GetName());
}

void GreaterLayer::Accept(ILayerVisitor& visitor) const
{
    visitor.VisitGreaterLayer(this, GetName());
}

} // namespace armnn
