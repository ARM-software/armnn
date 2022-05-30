//
// Copyright © 2017 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "PermuteLayer.hpp"

#include "LayerCloneBase.hpp"

#include <armnn/TypesUtils.hpp>

#include <armnnUtils/Permute.hpp>

#include <armnn/backends/WorkloadData.hpp>
#include <armnn/backends/WorkloadFactory.hpp>

namespace armnn
{

PermuteLayer::PermuteLayer(const PermuteDescriptor& param, const char* name)
    : LayerWithParameters(1, 1, LayerType::Permute, param, name)
{
}

std::unique_ptr<IWorkload> PermuteLayer::CreateWorkload(const IWorkloadFactory& factory) const
{
    PermuteQueueDescriptor descriptor;
    SetAdditionalInfo(descriptor);

    return factory.CreateWorkload(LayerType::Permute, descriptor, PrepInfoAndDesc(descriptor));
}

PermuteLayer* PermuteLayer::Clone(Graph& graph) const
{
    return CloneBase<PermuteLayer>(graph, m_Param, GetName());
}

std::vector<TensorShape> PermuteLayer::InferOutputShapes(const std::vector<TensorShape>& inputShapes) const
{
    ARMNN_ASSERT(inputShapes.size() == 1);
    const TensorShape& inShape = inputShapes[0];
    return std::vector<TensorShape> ({armnnUtils::Permuted(inShape, m_Param.m_DimMappings)});
}

void PermuteLayer::ValidateTensorShapesFromInputs()
{
    VerifyLayerConnections(1, CHECK_LOCATION());

    const TensorShape& outputShape = GetOutputSlot(0).GetTensorInfo().GetShape();

    VerifyShapeInferenceType(outputShape, m_ShapeInferenceMethod);

    auto inferredShapes = InferOutputShapes({ GetInputSlot(0).GetConnection()->GetTensorInfo().GetShape() });

    ARMNN_ASSERT(inferredShapes.size() == 1);

    ValidateAndCopyShape(outputShape, inferredShapes[0], m_ShapeInferenceMethod, "PermuteLayer");
}

void PermuteLayer::ExecuteStrategy(IStrategy& strategy) const
{
    strategy.ExecuteStrategy(this, GetParameters(), {}, GetName());
}

} // namespace armnn
