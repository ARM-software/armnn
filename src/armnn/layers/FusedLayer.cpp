//
// Copyright © 2023 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "FusedLayer.hpp"
#include "LayerCloneBase.hpp"

#include <armnn/backends/Workload.hpp>
#include <armnn/TypesUtils.hpp>

namespace armnn
{

FusedLayer::FusedLayer(const FusedDescriptor& param, const char* name)
    : LayerWithParameters(param.m_NumInputSlots, param.m_NumOutputSlots, LayerType::Fused, param, name)
{}

FusedLayer::~FusedLayer()
{}

FusedLayer* FusedLayer::Clone(Graph& graph) const
{
    FusedLayer* clonedLayer = CloneBase<FusedLayer>(graph, m_Param, GetName());
    clonedLayer->m_AdditionalInfoObject = const_cast<FusedLayer*>(this)->m_AdditionalInfoObject;
    return clonedLayer;
}

std::unique_ptr<IWorkload> FusedLayer::CreateWorkload(const armnn::IWorkloadFactory& factory) const
{
    FusedQueueDescriptor descriptor;
    SetAdditionalInfo(descriptor);

    return factory.CreateWorkload(LayerType::Fused, descriptor, PrepInfoAndDesc(descriptor));
}

void FusedLayer::ValidateTensorShapesFromInputs()
{
    // NOTE: since the FusedLayer is an internal layer created from a valid SubgraphView,
    // we do not need to validate its input shapes
}

void FusedLayer::ExecuteStrategy(IStrategy& strategy) const
{
    strategy.ExecuteStrategy(this, GetParameters(), {}, GetName());
}

} // namespace armnn
