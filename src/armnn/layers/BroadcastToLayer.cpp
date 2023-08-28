//
// Copyright © 2023 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "BroadcastToLayer.hpp"

#include "LayerCloneBase.hpp"

#include <armnn/TypesUtils.hpp>
#include <armnn/backends/WorkloadData.hpp>
#include <armnn/backends/WorkloadFactory.hpp>

namespace armnn
{

BroadcastToLayer::BroadcastToLayer(const BroadcastToDescriptor& param, const char* name)
    : LayerWithParameters(1, 1, LayerType::BroadcastTo, param, name)
{}

std::unique_ptr<IWorkload> BroadcastToLayer::CreateWorkload(const IWorkloadFactory& factory) const
{
    BroadcastToQueueDescriptor descriptor;
    SetAdditionalInfo(descriptor);

    return factory.CreateWorkload(LayerType::BroadcastTo, descriptor, PrepInfoAndDesc(descriptor));
}

BroadcastToLayer* BroadcastToLayer::Clone(armnn::Graph& graph) const
{
    return CloneBase<BroadcastToLayer>(graph, m_Param, GetName());
}

std::vector<TensorShape> BroadcastToLayer::InferOutputShapes(const std::vector<TensorShape>&) const
{
    return std::vector<TensorShape>({ m_Param.m_BroadcastToShape });
}

void BroadcastToLayer::ValidateTensorShapesFromInputs()
{
    VerifyLayerConnections(1, CHECK_LOCATION());

    const TensorShape &outputShape = GetOutputSlot(0).GetTensorInfo().GetShape();

    VerifyShapeInferenceType(outputShape, m_ShapeInferenceMethod);

    auto inferredShapes = outputShape;

    ValidateAndCopyShape(outputShape, inferredShapes, m_ShapeInferenceMethod, "BroadcastToLayer");
}

void BroadcastToLayer::ExecuteStrategy(IStrategy& strategy) const
{
    strategy.ExecuteStrategy(this, GetParameters(), {}, GetName());
}

} //namespace armnn
