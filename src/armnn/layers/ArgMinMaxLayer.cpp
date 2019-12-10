//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "ArgMinMaxLayer.hpp"
#include "LayerCloneBase.hpp"

#include <armnn/TypesUtils.hpp>

#include <armnnUtils/TensorUtils.hpp>

#include <backendsCommon/WorkloadData.hpp>
#include <backendsCommon/WorkloadFactory.hpp>

namespace armnn
{

ArgMinMaxLayer::ArgMinMaxLayer(const ArgMinMaxDescriptor& param, const char* name)
    : LayerWithParameters(1, 1, LayerType::ArgMinMax, param, name)
{
}

std::unique_ptr<IWorkload> ArgMinMaxLayer::CreateWorkload(const IWorkloadFactory& factory) const
{
    ArgMinMaxQueueDescriptor descriptor;
    return factory.CreateArgMinMax(descriptor, PrepInfoAndDesc(descriptor));
}

ArgMinMaxLayer* ArgMinMaxLayer::Clone(Graph& graph) const
{
    return CloneBase<ArgMinMaxLayer>(graph, m_Param, GetName());
}

std::vector<TensorShape> ArgMinMaxLayer::InferOutputShapes(const std::vector<TensorShape>& inputShapes) const
{
    BOOST_ASSERT(inputShapes.size() == 1);

    TensorShape inputShape = inputShapes[0];
    auto inputNumDimensions = inputShape.GetNumDimensions();

    auto axis = m_Param.m_Axis;
    auto unsignedAxis = armnnUtils::GetUnsignedAxis(inputNumDimensions, axis);

    BOOST_ASSERT(unsignedAxis <= inputNumDimensions);

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

    auto inferredShapes = InferOutputShapes({ GetInputSlot(0).GetConnection()->GetTensorInfo().GetShape() });

    BOOST_ASSERT(inferredShapes.size() == 1);

    ConditionalThrowIfNotEqual<LayerValidationException>(
            "ArgMinMaxLayer: TensorShape set on OutputSlot does not match the inferred shape.",
            GetOutputSlot(0).GetTensorInfo().GetShape(),
            inferredShapes[0]);
}

void ArgMinMaxLayer::Accept(ILayerVisitor& visitor) const
{
    visitor.VisitArgMinMaxLayer(this, GetParameters(), GetName());
}

} // namespace armnn
