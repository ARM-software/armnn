//
// Copyright © 2020 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//
#include "UnmapLayer.hpp"

#include "LayerCloneBase.hpp"

#include <armnn/TypesUtils.hpp>
#include <armnn/backends/WorkloadData.hpp>
#include <armnn/backends/WorkloadFactory.hpp>
#include <backendsCommon/UnmapWorkload.hpp>

namespace armnn
{

UnmapLayer::UnmapLayer(const char* name)
    : Layer(1, 0, LayerType::Unmap, name)
{
}

UnmapLayer* UnmapLayer::Clone(Graph& graph) const
{
    return CloneBase<UnmapLayer>(graph, GetName());
}

std::unique_ptr<IWorkload> UnmapLayer::CreateWorkload(const IWorkloadFactory& factory) const
{
    IgnoreUnused(factory);
    UnmapQueueDescriptor descriptor;
    SetAdditionalInfo(descriptor);

    //This is different from other workloads. Does not get created by the workload factory.
    return std::make_unique<UnmapWorkload>(descriptor, PrepInfoAndDesc(descriptor));
}

void UnmapLayer::ValidateTensorShapesFromInputs()
{
    // validates that the input is connected.
    VerifyLayerConnections(1, CHECK_LOCATION());
    ARMNN_ASSERT(GetNumOutputSlots() == 0);
}

ARMNN_NO_DEPRECATE_WARN_BEGIN
void UnmapLayer::Accept(ILayerVisitor& visitor) const
{
    IgnoreUnused(visitor);
    throw armnn::Exception("UnmapLayer should not appear in an input graph");
}
ARMNN_NO_DEPRECATE_WARN_END

} // namespace armnn
