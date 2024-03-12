//
// Copyright © 2020-2024 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "ElementwiseUnaryLayer.hpp"

#include "LayerCloneBase.hpp"

#include <armnn/backends/WorkloadData.hpp>
#include <armnn/backends/WorkloadFactory.hpp>

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
    return factory.CreateWorkload(LayerType::ElementwiseUnary, descriptor, PrepInfoAndDesc(descriptor));
}

ElementwiseUnaryLayer* ElementwiseUnaryLayer::Clone(Graph& graph) const
{
    return CloneBase<ElementwiseUnaryLayer>(graph, m_Param, GetName());
}

std::vector<TensorShape> ElementwiseUnaryLayer::InferOutputShapes(const std::vector<TensorShape>& inputShapes) const
{
    // Should return the shape of the input tensor
    if (inputShapes.size() != 1)
    {
        throw armnn::Exception("inputShapes' size is \"" + std::to_string(inputShapes.size()) +
                               "\" - should be \"1\".");
    }

    const TensorShape& input = inputShapes[0];

    return std::vector<TensorShape>({ input });
}

void ElementwiseUnaryLayer::ValidateTensorShapesFromInputs()
{
    VerifyLayerConnections(1, CHECK_LOCATION());

    const TensorShape& outputShape = GetOutputSlot(0).GetTensorInfo().GetShape();

    VerifyShapeInferenceType(outputShape, m_ShapeInferenceMethod);

    std::vector<TensorShape> inferredShapes = InferOutputShapes({
        GetInputSlot(0).GetTensorInfo().GetShape()});

    if (inferredShapes.size() != 1)
    {
        throw armnn::Exception("inferredShapes has "
                               + std::to_string(inferredShapes.size()) +
                               " elements - should only have 1.");
    }

    ValidateAndCopyShape(outputShape, inferredShapes[0], m_ShapeInferenceMethod, GetLayerTypeAsCString(GetType()));
}

void ElementwiseUnaryLayer::ExecuteStrategy(IStrategy& strategy) const
{
    strategy.ExecuteStrategy(this, GetParameters(), {}, GetName());
}

} // namespace armnn
