//
// Copyright © 2017 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "SubtractionLayer.hpp"

#include "LayerCloneBase.hpp"

#include <armnn/TypesUtils.hpp>
#include <armnn/backends/WorkloadData.hpp>
#include <armnn/backends/WorkloadFactory.hpp>

namespace armnn
{

SubtractionLayer::SubtractionLayer(const char* name)
    : ElementwiseBaseLayer(2, 1, LayerType::Subtraction, name)
{
}

std::unique_ptr<IWorkload> SubtractionLayer::CreateWorkload(const IWorkloadFactory& factory) const
{
    SubtractionQueueDescriptor descriptor;
    SetAdditionalInfo(descriptor);

    return factory.CreateWorkload(LayerType::Subtraction, descriptor, PrepInfoAndDesc(descriptor));
}

SubtractionLayer* SubtractionLayer::Clone(Graph& graph) const
{
    return CloneBase<SubtractionLayer>(graph, GetName());
}

void SubtractionLayer::ExecuteStrategy(IStrategy& strategy) const
{
    strategy.ExecuteStrategy(this, GetParameters(), {}, GetName());
}

} // namespace armnn
