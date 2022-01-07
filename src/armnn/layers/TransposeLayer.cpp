//
// Copyright © 2020 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "TransposeLayer.hpp"

#include "LayerCloneBase.hpp"

#include <armnn/TypesUtils.hpp>

#include <armnnUtils/Transpose.hpp>

#include <armnn/backends/WorkloadData.hpp>
#include <armnn/backends/WorkloadFactory.hpp>

namespace armnn
{

TransposeLayer::TransposeLayer(const TransposeDescriptor& param, const char* name)
    : LayerWithParameters(1, 1, LayerType::Transpose, param, name)
{
}

std::unique_ptr<IWorkload> TransposeLayer::CreateWorkload(const IWorkloadFactory& factory) const
{
    TransposeQueueDescriptor descriptor;
    SetAdditionalInfo(descriptor);

    return factory.CreateWorkload(LayerType::Transpose, descriptor, PrepInfoAndDesc(descriptor));
}

TransposeLayer* TransposeLayer::Clone(Graph& graph) const
{
    return CloneBase<TransposeLayer>(graph, m_Param, GetName());
}

std::vector<TensorShape> TransposeLayer::InferOutputShapes(const std::vector<TensorShape>& inputShapes) const
{
    ARMNN_ASSERT(inputShapes.size() == 1);
    const TensorShape& inShape = inputShapes[0];
    return std::vector<TensorShape> ({armnnUtils::TransposeTensorShape(inShape, m_Param.m_DimMappings)});
}

void TransposeLayer::ValidateTensorShapesFromInputs()
{
    VerifyLayerConnections(1, CHECK_LOCATION());

    const TensorShape& outputShape = GetOutputSlot(0).GetTensorInfo().GetShape();

    VerifyShapeInferenceType(outputShape, m_ShapeInferenceMethod);

    auto inferredShapes = InferOutputShapes({ GetInputSlot(0).GetConnection()->GetTensorInfo().GetShape() });

    ARMNN_ASSERT(inferredShapes.size() == 1);

    ValidateAndCopyShape(outputShape, inferredShapes[0], m_ShapeInferenceMethod, "TransposeLayer");
}

ARMNN_NO_DEPRECATE_WARN_BEGIN
void TransposeLayer::Accept(ILayerVisitor& visitor) const
{
    visitor.VisitTransposeLayer(this, GetParameters(), GetName());
}
ARMNN_NO_DEPRECATE_WARN_END

} // namespace armnn
