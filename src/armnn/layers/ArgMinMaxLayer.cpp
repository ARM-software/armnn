//
// Copyright © 2017 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "ArgMinMaxLayer.hpp"
#include "LayerCloneBase.hpp"

#include <armnn/TypesUtils.hpp>

#include <armnnUtils/TensorUtils.hpp>

#include <armnn/backends/WorkloadData.hpp>
#include <armnn/backends/WorkloadFactory.hpp>

namespace armnn
{

ArgMinMaxLayer::ArgMinMaxLayer(const ArgMinMaxDescriptor& param, const char* name)
    : LayerWithParameters(1, 1, LayerType::ArgMinMax, param, name)
{
}

std::unique_ptr<IWorkload> ArgMinMaxLayer::CreateWorkload(const IWorkloadFactory& factory) const
{
    ArgMinMaxQueueDescriptor descriptor;
    SetAdditionalInfo(descriptor);

    return factory.CreateWorkload(LayerType::ArgMinMax, descriptor, PrepInfoAndDesc(descriptor));
}

ArgMinMaxLayer* ArgMinMaxLayer::Clone(Graph& graph) const
{
    return CloneBase<ArgMinMaxLayer>(graph, m_Param, GetName());
}

std::vector<TensorShape> ArgMinMaxLayer::InferOutputShapes(const std::vector<TensorShape>& inputShapes) const
{
    ARMNN_ASSERT(inputShapes.size() == 1);

    TensorShape inputShape = inputShapes[0];
    auto inputNumDimensions = inputShape.GetNumDimensions();

    auto axis = m_Param.m_Axis;
    auto unsignedAxis = armnnUtils::GetUnsignedAxis(inputNumDimensions, axis);

    ARMNN_ASSERT(unsignedAxis <= inputNumDimensions);

    // 1D input shape results in scalar output
    if (inputShape.GetNumDimensions() == 1)
    {
        std::vector<unsigned int> tensorDimensions(1, 1);
        TensorShape outputShape(1, tensorDimensions.data());

        return std::vector<TensorShape>({ outputShape });
    }

    std::vector<unsigned int> tensorDimensions(inputNumDimensions - 1, 0);
    for (unsigned int i = 0; i < unsignedAxis; ++i)
    {
        tensorDimensions[i] = inputShape[i];
    }

    for (unsigned int i = unsignedAxis + 1; i < inputNumDimensions; ++i)
    {
        tensorDimensions[i - 1] = inputShape[i];
    }

    TensorShape outputShape = TensorShape(inputNumDimensions - 1, tensorDimensions.data());

    return std::vector<TensorShape>({ outputShape });
}

void ArgMinMaxLayer::ValidateTensorShapesFromInputs()
{
    VerifyLayerConnections(1, CHECK_LOCATION());

    const TensorShape& outputShape = GetOutputSlot(0).GetTensorInfo().GetShape();

    VerifyShapeInferenceType(outputShape, m_ShapeInferenceMethod);

    auto inferredShapes = InferOutputShapes({ GetInputSlot(0).GetConnection()->GetTensorInfo().GetShape() });

    ARMNN_ASSERT(inferredShapes.size() == 1);

    ValidateAndCopyShape(outputShape, inferredShapes[0], m_ShapeInferenceMethod, "ArgMinMaxLayer");
}

void ArgMinMaxLayer::ExecuteStrategy(IStrategy& strategy) const
{
    strategy.ExecuteStrategy(this, GetParameters(), {}, GetName());
}

} // namespace armnn
