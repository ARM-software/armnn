//
// Copyright © 2017 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//
#include "MergeLayer.hpp"

#include "LayerCloneBase.hpp"

#include <backendsCommon/WorkloadData.hpp>
#include <backendsCommon/WorkloadFactory.hpp>

namespace armnn
{

MergeLayer::MergeLayer(const char* name)
    : Layer(2, 1, LayerType::Merge, name)
{}

std::unique_ptr<IWorkload> MergeLayer::CreateWorkload(const IWorkloadFactory& factory) const
{
    IgnoreUnused(factory);
    return nullptr;
}

MergeLayer* MergeLayer::Clone(Graph& graph) const
{
    return CloneBase<MergeLayer>(graph, GetName());
}

void MergeLayer::ValidateTensorShapesFromInputs()
{
    VerifyLayerConnections(2, CHECK_LOCATION());

    const TensorShape& outputShape = GetOutputSlot(0).GetTensorInfo().GetShape();

    VerifyShapeInferenceType(outputShape, m_ShapeInferenceMethod);

    std::vector<TensorShape> inferredShapes = InferOutputShapes({
        GetInputSlot(0).GetConnection()->GetTensorInfo().GetShape(),
        GetInputSlot(1).GetConnection()->GetTensorInfo().GetShape(),
    });

    ARMNN_ASSERT(inferredShapes.size() == 1);

    ValidateAndCopyShape(outputShape, inferredShapes[0], m_ShapeInferenceMethod, "MergeLayer");
}

std::vector<TensorShape> MergeLayer::InferOutputShapes(const std::vector<TensorShape>& inputShapes) const
{
    ARMNN_ASSERT(inputShapes.size() == 2);

    ConditionalThrowIfNotEqual<LayerValidationException>(
        "MergeLayer: TensorShapes set on inputs do not match",
        inputShapes[0],
        inputShapes[1]
    );

    return {inputShapes[0]};
}

void MergeLayer::Accept(ILayerVisitor& visitor) const
{
    visitor.VisitMergeLayer(this, GetName());
}

} // namespace armnn
