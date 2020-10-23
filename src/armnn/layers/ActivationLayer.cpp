//
// Copyright © 2017 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//
#include "ActivationLayer.hpp"
#include "LayerCloneBase.hpp"

#include <armnn/TypesUtils.hpp>
#include <backendsCommon/WorkloadData.hpp>
#include <backendsCommon/WorkloadFactory.hpp>

namespace armnn
{

ActivationLayer::ActivationLayer(const ActivationDescriptor& param, const char* name)
    : LayerWithParameters(1, 1, LayerType::Activation, param, name)
{
}

std::unique_ptr<IWorkload> ActivationLayer::CreateWorkload(const IWorkloadFactory& factory) const
{
    ActivationQueueDescriptor descriptor;
    SetAdditionalInfo(descriptor);

    return factory.CreateActivation(descriptor, PrepInfoAndDesc(descriptor));
}

ActivationLayer* ActivationLayer::Clone(Graph& graph) const
{
    return CloneBase<ActivationLayer>(graph, m_Param, GetName());
}

void ActivationLayer::ValidateTensorShapesFromInputs()
{
    VerifyLayerConnections(1, CHECK_LOCATION());

    const TensorShape& outputShape = GetOutputSlot(0).GetTensorInfo().GetShape();

    VerifyShapeInferenceType(outputShape, m_ShapeInferenceMethod);

    auto inferredShapes = InferOutputShapes({ GetInputSlot(0).GetConnection()->GetTensorInfo().GetShape() });

    ARMNN_ASSERT(inferredShapes.size() == 1);

    ValidateAndCopyShape(outputShape, inferredShapes[0], m_ShapeInferenceMethod, "ActivationLayer");
}

void ActivationLayer::Accept(ILayerVisitor& visitor) const
{
    visitor.VisitActivationLayer(this, GetParameters(), GetName());
}

} // namespace armnn
