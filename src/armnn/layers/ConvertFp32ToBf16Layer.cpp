//
// Copyright © 2020 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "ConvertFp32ToBf16Layer.hpp"
#include "LayerCloneBase.hpp"

#include <armnn/TypesUtils.hpp>

#include <armnn/backends/WorkloadData.hpp>
#include <armnn/backends/WorkloadFactory.hpp>

namespace armnn
{

ConvertFp32ToBf16Layer::ConvertFp32ToBf16Layer(const char* name)
    : Layer(1, 1, LayerType::ConvertFp32ToBf16, name)
{
}

std::unique_ptr<IWorkload> ConvertFp32ToBf16Layer::CreateWorkload(const IWorkloadFactory& factory) const
{
    ConvertFp32ToBf16QueueDescriptor descriptor;
    SetAdditionalInfo(descriptor);

    return factory.CreateWorkload(LayerType::ConvertFp32ToBf16, descriptor, PrepInfoAndDesc(descriptor));
}

ConvertFp32ToBf16Layer* ConvertFp32ToBf16Layer::Clone(Graph& graph) const
{
    return CloneBase<ConvertFp32ToBf16Layer>(graph, GetName());
}

void ConvertFp32ToBf16Layer::ValidateTensorShapesFromInputs()
{

    VerifyLayerConnections(1, CHECK_LOCATION());

    const TensorShape& outputShape = GetOutputSlot(0).GetTensorInfo().GetShape();

    VerifyShapeInferenceType(outputShape, m_ShapeInferenceMethod);

    auto inferredShapes = InferOutputShapes({ GetInputSlot(0).GetConnection()->GetTensorInfo().GetShape() });

    ARMNN_ASSERT(inferredShapes.size() == 1);

    ValidateAndCopyShape(outputShape, inferredShapes[0], m_ShapeInferenceMethod, "LayerName");
}

ARMNN_NO_DEPRECATE_WARN_BEGIN
void ConvertFp32ToBf16Layer::Accept(ILayerVisitor& visitor) const
{
    // these conversion layers are only inserted by the
    // optimizer and so will never be in an input graph.
    IgnoreUnused(visitor);
    throw armnn::Exception("ConvertFp32ToBf16Layer should never appear in an input graph");
}
ARMNN_NO_DEPRECATE_WARN_END

} // namespace armnn
