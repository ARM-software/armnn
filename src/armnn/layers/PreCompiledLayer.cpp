//
// Copyright © 2017 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "PreCompiledLayer.hpp"

#include "LayerCloneBase.hpp"

#include <armnn/backends/Workload.hpp>

#include <armnn/TypesUtils.hpp>

namespace armnn
{

PreCompiledLayer::PreCompiledLayer(const PreCompiledDescriptor& param, const char* name)
    : LayerWithParameters(param.m_NumInputSlots, param.m_NumOutputSlots, LayerType::PreCompiled, param, name)
{}

PreCompiledLayer::~PreCompiledLayer()
{}

PreCompiledLayer* PreCompiledLayer::Clone(Graph& graph) const
{
    PreCompiledLayer* clone = CloneBase<PreCompiledLayer>(graph, m_Param, GetName());
    clone->m_PreCompiledObject = const_cast<PreCompiledLayer*>(this)->m_PreCompiledObject;
    return clone;
}

std::unique_ptr<IWorkload> PreCompiledLayer::CreateWorkload(const armnn::IWorkloadFactory& factory) const
{
    PreCompiledQueueDescriptor descriptor;
    descriptor.m_PreCompiledObject = m_PreCompiledObject.get();
    SetAdditionalInfo(descriptor);

    return factory.CreateWorkload(LayerType::PreCompiled, descriptor, PrepInfoAndDesc(descriptor));
}

void PreCompiledLayer::ValidateTensorShapesFromInputs()
{

    // NOTE: since the PreCompiledLayer is an internal layer created from a valid SubgraphView,
    // we do not need to validate its input shapes
}

void PreCompiledLayer::SetPreCompiledObject(PreCompiledObjectPtr preCompiledObject)
{
    m_PreCompiledObject = std::move(preCompiledObject);
}

void PreCompiledLayer::ExecuteStrategy(IStrategy& strategy) const
{
    strategy.ExecuteStrategy(this, GetParameters(), {}, GetName());
}

} // namespace armnn
