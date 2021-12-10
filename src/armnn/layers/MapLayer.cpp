//
// Copyright © 2020 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//
#include "MapLayer.hpp"

#include "LayerCloneBase.hpp"

#include <armnn/TypesUtils.hpp>
#include <armnn/backends/WorkloadData.hpp>
#include <armnn/backends/WorkloadFactory.hpp>
#include <backendsCommon/MapWorkload.hpp>

namespace armnn
{

MapLayer::MapLayer(const char* name)
    : Layer(1, 0, LayerType::Map, name)
{
}

MapLayer* MapLayer::Clone(Graph& graph) const
{
    return CloneBase<MapLayer>(graph, GetName());
}

std::unique_ptr<IWorkload> MapLayer::CreateWorkload(const IWorkloadFactory& factory) const
{
    IgnoreUnused(factory);
    MapQueueDescriptor descriptor;
    SetAdditionalInfo(descriptor);

    //This is different from other workloads. Does not get created by the workload factory.
    return std::make_unique<MapWorkload>(descriptor, PrepInfoAndDesc(descriptor));
}

void MapLayer::ValidateTensorShapesFromInputs()
{
    // validates that the input is connected.
    VerifyLayerConnections(1, CHECK_LOCATION());
    ARMNN_ASSERT(GetNumOutputSlots() == 0);
}

ARMNN_NO_DEPRECATE_WARN_BEGIN
void MapLayer::Accept(ILayerVisitor& visitor) const
{
    IgnoreUnused(visitor);
    throw armnn::Exception("MapLayer should not appear in an input graph");
}
ARMNN_NO_DEPRECATE_WARN_END

} // namespace armnn
