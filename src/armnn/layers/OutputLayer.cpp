//
// Copyright © 2017 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//
#include "OutputLayer.hpp"

#include "LayerCloneBase.hpp"

#include <armnn/utility/IgnoreUnused.hpp>
#include <armnn/backends/WorkloadData.hpp>
#include <armnn/backends/WorkloadFactory.hpp>

namespace armnn
{

OutputLayer::OutputLayer(LayerBindingId id, const char* name)
    : BindableLayer(1, 0, LayerType::Output, name, id)
{
}

std::unique_ptr<IWorkload> OutputLayer::CreateWorkload(const IWorkloadFactory& factory) const
{
    IgnoreUnused(factory);
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

ARMNN_NO_DEPRECATE_WARN_BEGIN
void OutputLayer::Accept(ILayerVisitor& visitor) const
{
    visitor.VisitOutputLayer(this, GetBindingId(), GetName());
}
ARMNN_NO_DEPRECATE_WARN_END

} // namespace armnn
