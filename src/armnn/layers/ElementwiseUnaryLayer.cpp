//
// Copyright © 2019 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "ElementwiseUnaryLayer.hpp"

#include "LayerCloneBase.hpp"

#include <backendsCommon/WorkloadData.hpp>
#include <backendsCommon/WorkloadFactory.hpp>

#include <algorithm>

namespace armnn
{

ElementwiseUnaryLayer::ElementwiseUnaryLayer(const ElementwiseUnaryDescriptor& param, const char* name)
    : LayerWithParameters(1, 1, LayerType::ElementwiseUnary, param, name)
{
}

std::unique_ptr<IWorkload> ElementwiseUnaryLayer::CreateWorkload(const IWorkloadFactory& factory) const
{
    ElementwiseUnaryQueueDescriptor descriptor;
    return factory.CreateElementwiseUnary(descriptor, PrepInfoAndDesc(descriptor));
}

ElementwiseUnaryLayer* ElementwiseUnaryLayer::Clone(Graph& graph) const
{
    return CloneBase<ElementwiseUnaryLayer>(graph, m_Param, GetName());
}

std::vector<TensorShape> ElementwiseUnaryLayer::InferOutputShapes(const std::vector<TensorShape>& inputShapes) const
{
    // Should return the shape of the input tensor
    BOOST_ASSERT(inputShapes.size() == 1);
    const TensorShape& input = inputShapes[0];

    return std::vector<TensorShape>({ input });
}

void ElementwiseUnaryLayer::ValidateTensorShapesFromInputs()
{
    VerifyLayerConnections(1, CHECK_LOCATION());

    std::vector<TensorShape> inferredShapes = InferOutputShapes({
        GetInputSlot(0).GetConnection()->GetTensorInfo().GetShape()});
    BOOST_ASSERT(inferredShapes.size() == 1);

    ConditionalThrowIfNotEqual<LayerValidationException>(
        "ElementwiseUnaryLayer: TensorShape set on OutputSlot[0] does not match the inferred shape.",
        GetOutputSlot(0).GetTensorInfo().GetShape(),
        inferredShapes[0]);
}

void ElementwiseUnaryLayer::Accept(ILayerVisitor& visitor) const
{
    visitor.VisitElementwiseUnaryLayer(this, GetParameters(), GetName());
}

} // namespace armnn
