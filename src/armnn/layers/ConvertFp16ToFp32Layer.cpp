//
// Copyright © 2017 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "ConvertFp16ToFp32Layer.hpp"
#include "LayerCloneBase.hpp"

#include <armnn/TypesUtils.hpp>

#include <armnn/backends/WorkloadData.hpp>
#include <armnn/backends/WorkloadFactory.hpp>

namespace armnn
{

ConvertFp16ToFp32Layer::ConvertFp16ToFp32Layer(const char* name)
    : Layer(1, 1, LayerType::ConvertFp16ToFp32, name)
{
}

std::unique_ptr<IWorkload> ConvertFp16ToFp32Layer::CreateWorkload(const IWorkloadFactory& factory) const
{
    ConvertFp16ToFp32QueueDescriptor descriptor;
    SetAdditionalInfo(descriptor);

    return factory.CreateWorkload(LayerType::ConvertFp16ToFp32, descriptor, PrepInfoAndDesc(descriptor));
}

ConvertFp16ToFp32Layer* ConvertFp16ToFp32Layer::Clone(Graph& graph) const
{
    return CloneBase<ConvertFp16ToFp32Layer>(graph, GetName());
}

void ConvertFp16ToFp32Layer::ValidateTensorShapesFromInputs()
{
    VerifyLayerConnections(1, CHECK_LOCATION());

    const TensorShape& outputShape = GetOutputSlot(0).GetTensorInfo().GetShape();

    VerifyShapeInferenceType(outputShape, m_ShapeInferenceMethod);

    auto inferredShapes = InferOutputShapes({ GetInputSlot(0).GetConnection()->GetTensorInfo().GetShape() });

    ARMNN_ASSERT(inferredShapes.size() == 1);

    ValidateAndCopyShape(outputShape, inferredShapes[0], m_ShapeInferenceMethod, "ConvertFp16ToFp32Layer");
}

ARMNN_NO_DEPRECATE_WARN_BEGIN
void ConvertFp16ToFp32Layer::Accept(ILayerVisitor& visitor) const
{
    // these conversion layers are only inserted by the
    // optimizer and so will never be in an input graph.
    IgnoreUnused(visitor);
    throw armnn::Exception("ConvertFp16ToFp32Layer should never appear in an input graph");
}
ARMNN_NO_DEPRECATE_WARN_END

} // namespace armnn
