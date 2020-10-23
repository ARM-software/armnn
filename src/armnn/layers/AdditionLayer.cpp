//
// Copyright © 2017 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "AdditionLayer.hpp"

#include "LayerCloneBase.hpp"

#include <armnn/TypesUtils.hpp>
#include <backendsCommon/WorkloadData.hpp>
#include <backendsCommon/WorkloadFactory.hpp>

namespace armnn
{

AdditionLayer::AdditionLayer(const char* name)
    : ElementwiseBaseLayer(2, 1, LayerType::Addition, name)
{
}

std::unique_ptr<IWorkload> AdditionLayer::CreateWorkload(const IWorkloadFactory& factory) const
{
    AdditionQueueDescriptor descriptor;
    SetAdditionalInfo(descriptor);

    return factory.CreateAddition(descriptor, PrepInfoAndDesc(descriptor));
}

AdditionLayer* AdditionLayer::Clone(Graph& graph) const
{
    return CloneBase<AdditionLayer>(graph, GetName());
}

void AdditionLayer::Accept(ILayerVisitor& visitor) const
{
    visitor.VisitAdditionLayer(this, GetName());
}

} // namespace armnn
