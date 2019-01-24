//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "PadLayer.hpp"
#include "LayerCloneBase.hpp"

#include <backendsCommon/CpuTensorHandle.hpp>
#include <backendsCommon/WorkloadData.hpp>
#include <backendsCommon/WorkloadFactory.hpp>

#include <cstring>

namespace armnn
{

PadLayer::PadLayer(const armnn::PadDescriptor& param, const char* name)
    : LayerWithParameters(1, 1, LayerType::Pad, param, name)
{}

std::unique_ptr<IWorkload> PadLayer::CreateWorkload(const armnn::Graph& graph,
                                                    const armnn::IWorkloadFactory& factory) const
{
    PadQueueDescriptor descriptor;
    descriptor.m_Parameters.m_PadList = m_Param.m_PadList;

    return factory.CreatePad(descriptor, PrepInfoAndDesc(descriptor, graph));
}

PadLayer* PadLayer::Clone(Graph& graph) const
{
    auto layer = CloneBase<PadLayer>(graph, m_Param, GetName());

    layer->m_Param.m_PadList = m_Param.m_PadList;

    return std::move(layer);
}

void PadLayer::ValidateTensorShapesFromInputs()
{
    return;
}

void PadLayer::Accept(ILayerVisitor& visitor) const
{
    visitor.VisitPadLayer(this, GetParameters(), GetName());
}

} // namespace armnn