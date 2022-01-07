//
// Copyright © 2017 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//
#include "L2NormalizationLayer.hpp"

#include "LayerCloneBase.hpp"

#include <armnn/TypesUtils.hpp>
#include <armnn/backends/WorkloadData.hpp>
#include <armnn/backends/WorkloadFactory.hpp>

namespace armnn
{

L2NormalizationLayer::L2NormalizationLayer(const L2NormalizationDescriptor& param, const char* name)
    : LayerWithParameters(1, 1, LayerType::L2Normalization, param, name)
{
}

std::unique_ptr<IWorkload> L2NormalizationLayer::CreateWorkload(const IWorkloadFactory& factory) const
{
    L2NormalizationQueueDescriptor descriptor;
    SetAdditionalInfo(descriptor);

    return factory.CreateWorkload(LayerType::L2Normalization, descriptor, PrepInfoAndDesc(descriptor));
}

L2NormalizationLayer* L2NormalizationLayer::Clone(Graph& graph) const
{
    return CloneBase<L2NormalizationLayer>(graph, m_Param, GetName());
}

void L2NormalizationLayer::ValidateTensorShapesFromInputs()
{
    VerifyLayerConnections(1, CHECK_LOCATION());

    const TensorShape& outputShape = GetOutputSlot(0).GetTensorInfo().GetShape();

    VerifyShapeInferenceType(outputShape, m_ShapeInferenceMethod);

    auto inferredShapes = InferOutputShapes({ GetInputSlot(0).GetConnection()->GetTensorInfo().GetShape() });

    ARMNN_ASSERT(inferredShapes.size() == 1);

    ValidateAndCopyShape(outputShape, inferredShapes[0], m_ShapeInferenceMethod, "L2NormalizationLayer");
}

ARMNN_NO_DEPRECATE_WARN_BEGIN
void L2NormalizationLayer::Accept(ILayerVisitor& visitor) const
{
    visitor.VisitL2NormalizationLayer(this, GetParameters(), GetName());
}
ARMNN_NO_DEPRECATE_WARN_END

} // namespace armnn
