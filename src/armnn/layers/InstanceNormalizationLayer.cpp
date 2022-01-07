//
// Copyright © 2019 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//
#include "InstanceNormalizationLayer.hpp"

#include "LayerCloneBase.hpp"

#include <armnn/TypesUtils.hpp>
#include <armnn/backends/WorkloadData.hpp>
#include <armnn/backends/WorkloadFactory.hpp>

namespace armnn
{

InstanceNormalizationLayer::InstanceNormalizationLayer(const InstanceNormalizationDescriptor& param, const char* name)
    : LayerWithParameters(1, 1, LayerType::InstanceNormalization, param, name)
{
}

std::unique_ptr<IWorkload> InstanceNormalizationLayer::CreateWorkload(const IWorkloadFactory& factory) const
{
    InstanceNormalizationQueueDescriptor descriptor;
    SetAdditionalInfo(descriptor);

    return factory.CreateWorkload(LayerType::InstanceNormalization, descriptor, PrepInfoAndDesc(descriptor));
}

InstanceNormalizationLayer* InstanceNormalizationLayer::Clone(Graph& graph) const
{
    return CloneBase<InstanceNormalizationLayer>(graph, m_Param, GetName());
}

void InstanceNormalizationLayer::ValidateTensorShapesFromInputs()
{
    VerifyLayerConnections(1, CHECK_LOCATION());

    const TensorShape& outputShape = GetOutputSlot(0).GetTensorInfo().GetShape();

    VerifyShapeInferenceType(outputShape, m_ShapeInferenceMethod);

    auto inferredShapes = InferOutputShapes({ GetInputSlot(0).GetConnection()->GetTensorInfo().GetShape() });

    ARMNN_ASSERT(inferredShapes.size() == 1);

    ValidateAndCopyShape(outputShape, inferredShapes[0], m_ShapeInferenceMethod, "InstanceNormalizationLayer");
}

ARMNN_NO_DEPRECATE_WARN_BEGIN
void InstanceNormalizationLayer::Accept(ILayerVisitor& visitor) const
{
    visitor.VisitInstanceNormalizationLayer(this, GetParameters(), GetName());
}
ARMNN_NO_DEPRECATE_WARN_END

} // namespace armnn
