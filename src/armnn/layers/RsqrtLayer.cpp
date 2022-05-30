//
// Copyright © 2017 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "RsqrtLayer.hpp"

#include "LayerCloneBase.hpp"

#include <armnn/TypesUtils.hpp>
#include <armnn/backends/WorkloadData.hpp>
#include <armnn/backends/WorkloadFactory.hpp>

namespace armnn
{

RsqrtLayer::RsqrtLayer(const char* name)
    : Layer(1, 1, LayerType::Rsqrt, name)
{
}

std::unique_ptr<IWorkload> RsqrtLayer::CreateWorkload(const IWorkloadFactory& factory) const
{
    ElementwiseUnaryQueueDescriptor descriptor;
    descriptor.m_Parameters.m_Operation = UnaryOperation::Rsqrt;
    SetAdditionalInfo(descriptor);

    return factory.CreateWorkload(LayerType::ElementwiseUnary, descriptor, PrepInfoAndDesc(descriptor));
}

RsqrtLayer* RsqrtLayer::Clone(Graph& graph) const
{
    return CloneBase<RsqrtLayer>(graph, GetName());
}

void RsqrtLayer::ValidateTensorShapesFromInputs()
{
    VerifyLayerConnections(1, CHECK_LOCATION());

    const TensorShape& outputShape = GetOutputSlot(0).GetTensorInfo().GetShape();

    VerifyShapeInferenceType(outputShape, m_ShapeInferenceMethod);

    auto inferredShapes = InferOutputShapes({ GetInputSlot(0).GetConnection()->GetTensorInfo().GetShape() });

    ARMNN_ASSERT(inferredShapes.size() == 1);

    ValidateAndCopyShape(outputShape, inferredShapes[0], m_ShapeInferenceMethod, "RsqrtLayer");
}

void RsqrtLayer::ExecuteStrategy(IStrategy& strategy) const
{
    strategy.ExecuteStrategy(this, GetParameters(), {}, GetName());
}

} // namespace armnn