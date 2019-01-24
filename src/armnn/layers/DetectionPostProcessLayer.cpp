//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "DetectionPostProcessLayer.hpp"

#include "LayerCloneBase.hpp"

#include <armnn/TypesUtils.hpp>
#include <backendsCommon/WorkloadData.hpp>
#include <backendsCommon/WorkloadFactory.hpp>

namespace armnn
{

DetectionPostProcessLayer::DetectionPostProcessLayer(const DetectionPostProcessDescriptor& param, const char* name)
    : LayerWithParameters(2, 4, LayerType::DetectionPostProcess, param, name)
{
}

std::unique_ptr<IWorkload> DetectionPostProcessLayer::CreateWorkload(const armnn::Graph& graph,
                                                                     const armnn::IWorkloadFactory& factory) const
{
    DetectionPostProcessQueueDescriptor descriptor;
    return factory.CreateDetectionPostProcess(descriptor, PrepInfoAndDesc(descriptor, graph));
}

DetectionPostProcessLayer* DetectionPostProcessLayer::Clone(Graph& graph) const
{
    return CloneBase<DetectionPostProcessLayer>(graph, m_Param, GetName());
}

void DetectionPostProcessLayer::ValidateTensorShapesFromInputs()
{
    VerifyLayerConnections(2, CHECK_LOCATION());
}

void DetectionPostProcessLayer::Accept(ILayerVisitor& visitor) const
{
    visitor.VisitDetectionPostProcessLayer(this, GetParameters(), GetName());
}

} // namespace armnn
