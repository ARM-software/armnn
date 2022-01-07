//
// Copyright © 2017 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//
#include "NormalizationLayer.hpp"

#include "LayerCloneBase.hpp"

#include <armnn/TypesUtils.hpp>
#include <armnn/backends/WorkloadData.hpp>
#include <armnn/backends/WorkloadFactory.hpp>

namespace armnn
{

NormalizationLayer::NormalizationLayer(const NormalizationDescriptor& param, const char* name)
    : LayerWithParameters(1, 1, LayerType::Normalization, param, name)
{
}

std::unique_ptr<IWorkload> NormalizationLayer::CreateWorkload(const IWorkloadFactory& factory) const
{
    NormalizationQueueDescriptor descriptor;
    SetAdditionalInfo(descriptor);

    return factory.CreateWorkload(LayerType::Normalization, descriptor, PrepInfoAndDesc(descriptor));
}

NormalizationLayer* NormalizationLayer::Clone(Graph& graph) const
{
    return CloneBase<NormalizationLayer>(graph, m_Param, GetName());
}

void NormalizationLayer::ValidateTensorShapesFromInputs()
{
    VerifyLayerConnections(1, CHECK_LOCATION());

    const TensorShape& outputShape = GetOutputSlot(0).GetTensorInfo().GetShape();

    VerifyShapeInferenceType(outputShape, m_ShapeInferenceMethod);

    auto inferredShapes = InferOutputShapes({ GetInputSlot(0).GetConnection()->GetTensorInfo().GetShape() });

    ARMNN_ASSERT(inferredShapes.size() == 1);

    ValidateAndCopyShape(outputShape, inferredShapes[0], m_ShapeInferenceMethod, "NormalizationLayer");
}

ARMNN_NO_DEPRECATE_WARN_BEGIN
void NormalizationLayer::Accept(ILayerVisitor& visitor) const
{
    visitor.VisitNormalizationLayer(this, GetParameters(), GetName());
}
ARMNN_NO_DEPRECATE_WARN_END

} // namespace armnn
