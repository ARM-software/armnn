//
// Copyright © 2017-2023 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "MaximumLayer.hpp"

#include "LayerCloneBase.hpp"

#include <armnn/backends/WorkloadData.hpp>
#include <armnn/backends/WorkloadFactory.hpp>

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

    return factory.CreateWorkload(LayerType::Maximum, descriptor, PrepInfoAndDesc(descriptor));
}

ARMNN_NO_DEPRECATE_WARN_BEGIN
MaximumLayer* MaximumLayer::Clone(Graph& graph) const
{
    return CloneBase<MaximumLayer>(graph, GetName());
}
ARMNN_NO_DEPRECATE_WARN_END

void MaximumLayer::ExecuteStrategy(IStrategy& strategy) const
{
    strategy.ExecuteStrategy(this, GetParameters(), {}, GetName());
}

} // namespace armnn
