//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "GatherLayer.hpp"
#include "LayerCloneBase.hpp"

#include <armnn/TypesUtils.hpp>
#include <backendsCommon/WorkloadData.hpp>
#include <backendsCommon/WorkloadFactory.hpp>

namespace armnn
{

GatherLayer::GatherLayer(const char* name)
    : Layer(2, 1, LayerType::Gather, name)
{
}

std::unique_ptr<IWorkload> GatherLayer::CreateWorkload(const armnn::Graph& graph,
                                                       const armnn::IWorkloadFactory& factory) const
{
    GatherQueueDescriptor descriptor;
    return factory.CreateGather(descriptor, PrepInfoAndDesc(descriptor, graph));
}

GatherLayer* GatherLayer::Clone(Graph& graph) const
{
    return CloneBase<GatherLayer>(graph, GetName());
}

void GatherLayer::ValidateTensorShapesFromInputs()
{
}

} // namespace armnn
