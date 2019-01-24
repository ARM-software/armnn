//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//
#include "OutputLayer.hpp"

#include "LayerCloneBase.hpp"

#include <backendsCommon/WorkloadData.hpp>
#include <backendsCommon/WorkloadFactory.hpp>

namespace armnn
{

OutputLayer::OutputLayer(LayerBindingId id, const char* name)
    : BindableLayer(1, 0, LayerType::Output, name, id)
{
}

std::unique_ptr<IWorkload> OutputLayer::CreateWorkload(const Graph& graph, const IWorkloadFactory& factory) const
{
    return nullptr;
}

OutputLayer* OutputLayer::Clone(Graph& graph) const
{
    return CloneBase<OutputLayer>(graph, GetBindingId(), GetName());
}

void OutputLayer::ValidateTensorShapesFromInputs()
{
    // Just validates that the input is connected.
    ConditionalThrow<LayerValidationException>(GetInputSlot(0).GetConnection() != nullptr,
                                               "OutputLayer: Input slot must be connected.");
}

void OutputLayer::Accept(ILayerVisitor& visitor) const
{
    visitor.VisitOutputLayer(this, GetBindingId(), GetName());
}

} // namespace armnn
