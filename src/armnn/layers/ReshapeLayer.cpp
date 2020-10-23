//
// Copyright © 2017 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//
#include "ReshapeLayer.hpp"

#include "LayerCloneBase.hpp"

#include <armnn/utility/IgnoreUnused.hpp>
#include <armnn/TypesUtils.hpp>
#include <backendsCommon/WorkloadData.hpp>
#include <backendsCommon/WorkloadFactory.hpp>

namespace armnn
{

ReshapeLayer::ReshapeLayer(const ReshapeDescriptor& param, const char* name)
    : LayerWithParameters(1, 1, LayerType::Reshape, param, name)
{
}

std::unique_ptr<IWorkload> ReshapeLayer::CreateWorkload(const IWorkloadFactory& factory) const
{
    ReshapeQueueDescriptor descriptor;
    SetAdditionalInfo(descriptor);

    return factory.CreateReshape(descriptor, PrepInfoAndDesc(descriptor));
}

ReshapeLayer* ReshapeLayer::Clone(Graph& graph) const
{
    return CloneBase<ReshapeLayer>(graph, m_Param, GetName());
}

std::vector<TensorShape> ReshapeLayer::InferOutputShapes(const std::vector<TensorShape>& inputShapes) const
{
    IgnoreUnused(inputShapes);
    return std::vector<TensorShape>({ m_Param.m_TargetShape });
}

void ReshapeLayer::ValidateTensorShapesFromInputs()
{
    VerifyLayerConnections(1, CHECK_LOCATION());

    const TensorShape& outputShape = GetOutputSlot(0).GetTensorInfo().GetShape();

    VerifyShapeInferenceType(outputShape, m_ShapeInferenceMethod);

    auto inferredShapes = InferOutputShapes({ GetInputSlot(0).GetConnection()->GetTensorInfo().GetShape() });

    ARMNN_ASSERT(inferredShapes.size() == 1);

    ValidateAndCopyShape(outputShape, inferredShapes[0], m_ShapeInferenceMethod, "ReshapeLayer");
}

void ReshapeLayer::Accept(ILayerVisitor& visitor) const
{
    visitor.VisitReshapeLayer(this, GetParameters(), GetName());
}

} // namespace armnn
