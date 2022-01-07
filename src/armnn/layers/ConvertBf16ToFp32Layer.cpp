//
// Copyright © 2020 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "ConvertBf16ToFp32Layer.hpp"
#include "LayerCloneBase.hpp"

#include <armnn/TypesUtils.hpp>

#include <armnn/backends/WorkloadData.hpp>
#include <armnn/backends/WorkloadFactory.hpp>

namespace armnn
{

ConvertBf16ToFp32Layer::ConvertBf16ToFp32Layer(const char* name)
    : Layer(1, 1, LayerType::ConvertBf16ToFp32, name)
{
}

std::unique_ptr<IWorkload> ConvertBf16ToFp32Layer::CreateWorkload(const IWorkloadFactory& factory) const
{
    ConvertBf16ToFp32QueueDescriptor descriptor;
    SetAdditionalInfo(descriptor);

    return factory.CreateWorkload(LayerType::ConvertBf16ToFp32, descriptor, PrepInfoAndDesc(descriptor));
}

ConvertBf16ToFp32Layer* ConvertBf16ToFp32Layer::Clone(Graph& graph) const
{
    return CloneBase<ConvertBf16ToFp32Layer>(graph, GetName());
}

void ConvertBf16ToFp32Layer::ValidateTensorShapesFromInputs()
{
    VerifyLayerConnections(1, CHECK_LOCATION());

    const TensorShape& outputShape = GetOutputSlot(0).GetTensorInfo().GetShape();

    VerifyShapeInferenceType(outputShape, m_ShapeInferenceMethod);

    auto inferredShapes = InferOutputShapes({ GetInputSlot(0).GetConnection()->GetTensorInfo().GetShape() });

    ARMNN_ASSERT(inferredShapes.size() == 1);

    ValidateAndCopyShape(outputShape, inferredShapes[0], m_ShapeInferenceMethod, "ConvertBf16ToFp32Layer");
}

ARMNN_NO_DEPRECATE_WARN_BEGIN
void ConvertBf16ToFp32Layer::Accept(ILayerVisitor& visitor) const
{
    // these conversion layers are only inserted by the
    // optimizer and so will never be in an input graph.
    IgnoreUnused(visitor);
    throw armnn::Exception("ConvertBf16ToFp32Layer should never appear in an input graph");
}
ARMNN_NO_DEPRECATE_WARN_END

} // namespace armnn
