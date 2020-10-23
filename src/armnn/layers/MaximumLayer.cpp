//
// Copyright © 2017 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "MaximumLayer.hpp"

#include "LayerCloneBase.hpp"

#include <backendsCommon/WorkloadData.hpp>
#include <backendsCommon/WorkloadFactory.hpp>

namespace armnn
{

MaximumLayer::MaximumLayer(const char* name)
    : ElementwiseBaseLayer(2, 1, LayerType::Maximum, name)
{
}

std::unique_ptr<IWorkload> MaximumLayer::CreateWorkload(const IWorkloadFactory& factory) const
{
    MaximumQueueDescriptor descriptor;
    SetAdditionalInfo(descriptor);

    return factory.CreateMaximum(descriptor, PrepInfoAndDesc(descriptor));
}

MaximumLayer* MaximumLayer::Clone(Graph& graph) const
{
    return CloneBase<MaximumLayer>(graph, GetName());
}

void MaximumLayer::Accept(ILayerVisitor& visitor) const
{
    visitor.VisitMaximumLayer(this, GetName());
}

} // namespace armnn
