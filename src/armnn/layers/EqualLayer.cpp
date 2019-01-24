//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "EqualLayer.hpp"

#include "LayerCloneBase.hpp"

#include <armnn/TypesUtils.hpp>
#include <backendsCommon/WorkloadData.hpp>
#include <backendsCommon/WorkloadFactory.hpp>

namespace armnn
{

EqualLayer::EqualLayer(const char* name)
    : ElementwiseBaseLayer(2, 1, LayerType::Equal, name)
{
}

std::unique_ptr<IWorkload> EqualLayer::CreateWorkload(const Graph& graph,
                                                        const IWorkloadFactory& factory) const
{
    EqualQueueDescriptor descriptor;
    return factory.CreateEqual(descriptor, PrepInfoAndDesc(descriptor, graph));
}

EqualLayer* EqualLayer::Clone(Graph& graph) const
{
    return CloneBase<EqualLayer>(graph, GetName());
}

void EqualLayer::Accept(ILayerVisitor& visitor) const
{
    visitor.VisitEqualLayer(this, GetName());
}

} // namespace armnn
