//
// Copyright © 2017 Arm Ltd. All rights reserved.
// See LICENSE file in the project root for full license information.
//
#include "AdditionLayer.hpp"

#include "LayerCloneBase.hpp"

#include <armnn/TypesUtils.hpp>
#include <backends/WorkloadData.hpp>
#include <backends/WorkloadFactory.hpp>

namespace armnn
{

AdditionLayer::AdditionLayer(const char* name)
    : ArithmeticBaseLayer(2, 1, LayerType::Addition, name)
{
}

std::unique_ptr<IWorkload> AdditionLayer::CreateWorkload(const Graph& graph,
                                                         const IWorkloadFactory& factory) const
{
    AdditionQueueDescriptor descriptor;
    return factory.CreateAddition(descriptor, PrepInfoAndDesc(descriptor, graph));
}

AdditionLayer* AdditionLayer::Clone(Graph& graph) const
{
    return CloneBase<AdditionLayer>(graph, GetName());
}

} // namespace armnn
