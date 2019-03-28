//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//
#include "DequantizeLayer.hpp"

#include "LayerCloneBase.hpp"

#include <backendsCommon/WorkloadData.hpp>
#include <backendsCommon/WorkloadFactory.hpp>

namespace armnn
{

DequantizeLayer::DequantizeLayer(const char* name)
    : Layer(1, 1, LayerType::Dequantize, name)
{}

std::unique_ptr<IWorkload> DequantizeLayer::CreateWorkload(const Graph& graph,
                                                           const IWorkloadFactory& factory) const
{
    DequantizeQueueDescriptor descriptor;

    return factory.CreateDequantize(descriptor, PrepInfoAndDesc(descriptor, graph));
}

DequantizeLayer* DequantizeLayer::Clone(Graph& graph) const
{
    return CloneBase<DequantizeLayer>(graph, GetName());
}

void DequantizeLayer::ValidateTensorShapesFromInputs()
{
    VerifyLayerConnections(1, CHECK_LOCATION());

    std::vector<TensorShape> inferredShapes = InferOutputShapes({
        GetInputSlot(0).GetConnection()->GetTensorInfo().GetShape() });

    BOOST_ASSERT(inferredShapes.size() == 1);

    ConditionalThrowIfNotEqual<LayerValidationException>(
        "DequantizeLayer: TensorShape set on OutputSlot[0] does not match the inferred shape.",
        GetOutputSlot(0).GetTensorInfo().GetShape(),
        inferredShapes[0]);
}

void DequantizeLayer::Accept(ILayerVisitor& visitor) const
{
    visitor.VisitDequantizeLayer(this, GetName());
}

} // namespace armnn
