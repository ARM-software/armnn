//
// Copyright © 2017 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "SubtractionLayer.hpp"

#include "LayerCloneBase.hpp"

#include <armnn/TypesUtils.hpp>
#include <backendsCommon/WorkloadData.hpp>
#include <backendsCommon/WorkloadFactory.hpp>

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

    return factory.CreateSubtraction(descriptor, PrepInfoAndDesc(descriptor));
}

SubtractionLayer* SubtractionLayer::Clone(Graph& graph) const
{
    return CloneBase<SubtractionLayer>(graph, GetName());
}

void SubtractionLayer::Accept(ILayerVisitor& visitor) const
{
    visitor.VisitSubtractionLayer(this, GetName());
}

} // namespace armnn
