//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "QuantizeLayer.hpp"

#include "LayerCloneBase.hpp"

#include <armnn/ILayerVisitor.hpp>

namespace armnn
{

QuantizeLayer::QuantizeLayer(const char* name)
: Layer(1, 1, LayerType::Quantize, name)
{}

std::unique_ptr<IWorkload> QuantizeLayer::CreateWorkload(const Graph& graph,
                                                         const IWorkloadFactory& factory) const
{
    QuantizeQueueDescriptor descriptor;
    WorkloadInfo info = PrepInfoAndDesc(descriptor, graph);
    return factory.CreateQuantize(descriptor, info);
}

Layer* QuantizeLayer::Clone(Graph& graph) const
{
    QuantizeLayer* clone = CloneBase<QuantizeLayer>(graph, GetName());
    return clone;
}

void QuantizeLayer::ValidateTensorShapesFromInputs()
{
    VerifyLayerConnections(1, CHECK_LOCATION());

    auto inferredShapes = InferOutputShapes({ GetInputSlot(0).GetConnection()->GetTensorInfo().GetShape() });

    ConditionalThrowIfNotEqual<LayerValidationException>(
        "QuantizeLayer: TensorShape set on OutputSlot[0] does not match the inferred shape.",
        GetOutputSlot(0).GetTensorInfo().GetShape(),
        inferredShapes[0]);
}

void QuantizeLayer::Accept(ILayerVisitor& visitor) const
{
    visitor.VisitQuantizeLayer(this, GetName());
}

} //namespace armnn