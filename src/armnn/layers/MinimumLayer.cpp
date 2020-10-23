//
// Copyright © 2017 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "MinimumLayer.hpp"

#include "LayerCloneBase.hpp"

#include <armnn/TypesUtils.hpp>
#include <backendsCommon/WorkloadData.hpp>
#include <backendsCommon/WorkloadFactory.hpp>

namespace armnn
{

MinimumLayer::MinimumLayer(const char* name)
    : ElementwiseBaseLayer(2, 1, LayerType::Minimum, name)
{
}

std::unique_ptr<IWorkload> MinimumLayer::CreateWorkload(const IWorkloadFactory& factory) const
{
    MinimumQueueDescriptor descriptor;
    SetAdditionalInfo(descriptor);

    return factory.CreateMinimum(descriptor, PrepInfoAndDesc(descriptor));
}

MinimumLayer* MinimumLayer::Clone(Graph& graph) const
{
    return CloneBase<MinimumLayer>(graph, GetName());
}

void MinimumLayer::Accept(ILayerVisitor& visitor) const
{
    visitor.VisitMinimumLayer(this, GetName());
}

} // namespace armnn
