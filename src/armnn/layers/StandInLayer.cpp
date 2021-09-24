//
// Copyright © 2019 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "StandInLayer.hpp"
#include "LayerCloneBase.hpp"

namespace armnn
{

StandInLayer::StandInLayer(const StandInDescriptor& param, const char* name)
    : LayerWithParameters(param.m_NumInputs, param.m_NumOutputs, LayerType::StandIn, param, name)
{
}

std::unique_ptr<IWorkload> StandInLayer::CreateWorkload(const IWorkloadFactory& factory) const
{
    IgnoreUnused(factory);
    // This throws in the event that it's called. We would expect that any backend that
    // "claims" to support the StandInLayer type would actually substitute it with a PrecompiledLayer
    // during graph optimization. There is no interface on the IWorkloadFactory to create a StandInWorkload.
    throw Exception("Stand in layer does not support creating workloads");
}

StandInLayer* StandInLayer::Clone(Graph& graph) const
{
    return CloneBase<StandInLayer>(graph, m_Param, GetName());
}

std::vector<TensorShape> StandInLayer::InferOutputShapes(const std::vector<TensorShape>& inputShapes) const
{
    IgnoreUnused(inputShapes);
    throw Exception("Stand in layer does not support infering output shapes");
}

void StandInLayer::ValidateTensorShapesFromInputs()
{

    // Cannot validate this layer since no implementation details can be known by the framework
    // so do nothing here.
}

ARMNN_NO_DEPRECATE_WARN_BEGIN
void StandInLayer::Accept(ILayerVisitor& visitor) const
{
    visitor.VisitStandInLayer(this, GetParameters(), GetName());
}
ARMNN_NO_DEPRECATE_WARN_END
} // namespace armnn
