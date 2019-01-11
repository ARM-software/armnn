//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "PreCompiledLayer.hpp"

#include "LayerCloneBase.hpp"

#include "backendsCommon/Workload.hpp"

#include <armnn/TypesUtils.hpp>

namespace armnn
{

PreCompiledLayer::PreCompiledLayer(const PreCompiledDescriptor& param, const char* name)
    : LayerWithParameters(param.m_NumInputSlots, param.m_NumOutputSlots, LayerType::PreCompiled, param, name)
    , m_PreCompiledObject(nullptr)
{}

PreCompiledLayer::~PreCompiledLayer()
{}

PreCompiledLayer* PreCompiledLayer::Clone(Graph& graph) const
{
    PreCompiledLayer* clone = CloneBase<PreCompiledLayer>(graph, m_Param, GetName());
    clone->m_PreCompiledObject = this->m_PreCompiledObject;
    return clone;
}

std::unique_ptr<IWorkload> PreCompiledLayer::CreateWorkload(const armnn::Graph& graph,
                                                            const armnn::IWorkloadFactory& factory) const
{
    PreCompiledQueueDescriptor descriptor;
    descriptor.m_PreCompiledObject = m_PreCompiledObject;
    return factory.CreatePreCompiled(descriptor, PrepInfoAndDesc(descriptor, graph));
}

void PreCompiledLayer::ValidateTensorShapesFromInputs()
{
    // NOTE: since the PreCompiledLayer is an internal layer created from a valid SubGraph,
    // we do not need to validate its input shapes
}

std::shared_ptr<void> PreCompiledLayer::GetPreCompiledObject() const
{
    return m_PreCompiledObject;
}

void PreCompiledLayer::SetPreCompiledObject(const std::shared_ptr<void>& preCompiledObject)
{
    m_PreCompiledObject = preCompiledObject;
}

} // namespace armnn
