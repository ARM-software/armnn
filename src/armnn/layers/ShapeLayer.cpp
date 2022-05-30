//
// Copyright © 2021 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "ShapeLayer.hpp"

#include "LayerCloneBase.hpp"

#include <armnn/TypesUtils.hpp>
#include <armnn/utility/NumericCast.hpp>

#include <armnn/backends/WorkloadData.hpp>
#include <armnn/backends/WorkloadFactory.hpp>

namespace armnn
{

ShapeLayer::ShapeLayer(const char* name)
    : Layer(1, 1, LayerType::Shape, name)
{
}

std::unique_ptr<IWorkload> ShapeLayer::CreateWorkload(const IWorkloadFactory& factory) const
{
    ShapeQueueDescriptor descriptor;
    SetAdditionalInfo(descriptor);

    return factory.CreateWorkload(LayerType::Shape, descriptor, PrepInfoAndDesc(descriptor));
}

ShapeLayer* ShapeLayer::Clone(Graph& graph) const
{
    return CloneBase<ShapeLayer>(graph, GetName());
}

void ShapeLayer::ValidateTensorShapesFromInputs()
{
    VerifyLayerConnections(1, CHECK_LOCATION());

    const TensorShape& outputShape = GetOutputSlot(0).GetTensorInfo().GetShape();

    VerifyShapeInferenceType(outputShape, m_ShapeInferenceMethod);

    auto inferredShape = InferOutputShapes({ GetInputSlot(0).GetConnection()->GetTensorInfo().GetShape() });

    ARMNN_ASSERT(inferredShape.size() == 1);

    ValidateAndCopyShape(outputShape, inferredShape[0], m_ShapeInferenceMethod, "ShapeLayer");
}

std::vector<TensorShape> ShapeLayer::InferOutputShapes(const std::vector<TensorShape>& inputShapes) const
{
    IgnoreUnused(inputShapes);
    ARMNN_ASSERT(inputShapes.size() == 1);

    TensorShape outputShape({ inputShapes[0].GetNumDimensions()} );

    return std::vector<TensorShape>({ outputShape });
}


void ShapeLayer::ExecuteStrategy(IStrategy& strategy) const
{
    strategy.ExecuteStrategy(this, BaseDescriptor(), {}, GetName());
}

} // namespace armnn
