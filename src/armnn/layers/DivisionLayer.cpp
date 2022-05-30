//
// Copyright © 2017 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "DivisionLayer.hpp"

#include "LayerCloneBase.hpp"

#include <armnn/TypesUtils.hpp>
#include <armnn/backends/WorkloadData.hpp>
#include <armnn/backends/WorkloadFactory.hpp>

namespace armnn
{

DivisionLayer::DivisionLayer(const char* name)
    : ElementwiseBaseLayer(2, 1, LayerType::Division, name)
{
}

std::unique_ptr<IWorkload> DivisionLayer::CreateWorkload(const IWorkloadFactory& factory) const
{
    DivisionQueueDescriptor descriptor;
    SetAdditionalInfo(descriptor);

    return factory.CreateWorkload(LayerType::Division, descriptor, PrepInfoAndDesc(descriptor));
}

DivisionLayer* DivisionLayer::Clone(Graph& graph) const
{
    return CloneBase<DivisionLayer>(graph, GetName());
}

void DivisionLayer::ExecuteStrategy(IStrategy& strategy) const
{
    strategy.ExecuteStrategy(this, GetParameters(), {}, GetName());
}

} // namespace armnn
