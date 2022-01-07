//
// Copyright © 2019 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "LogSoftmaxLayer.hpp"

#include "LayerCloneBase.hpp"

#include <armnn/TypesUtils.hpp>

#include <armnn/backends/WorkloadData.hpp>
#include <armnn/backends/WorkloadFactory.hpp>

namespace armnn
{

LogSoftmaxLayer::LogSoftmaxLayer(const LogSoftmaxDescriptor &param, const char* name)
    : LayerWithParameters(1, 1, LayerType::LogSoftmax, param, name) {}

std::unique_ptr<IWorkload> LogSoftmaxLayer::CreateWorkload(const IWorkloadFactory& factory) const
{
    LogSoftmaxQueueDescriptor descriptor;
    SetAdditionalInfo(descriptor);

    return factory.CreateWorkload(LayerType::LogSoftmax, descriptor, PrepInfoAndDesc(descriptor));
}

LogSoftmaxLayer* LogSoftmaxLayer::Clone(Graph& graph) const
{
    return CloneBase<LogSoftmaxLayer>(graph, m_Param, GetName());
}

void LogSoftmaxLayer::ValidateTensorShapesFromInputs()
{
    VerifyLayerConnections(1, CHECK_LOCATION());

    const TensorShape& outputShape = GetOutputSlot(0).GetTensorInfo().GetShape();

    VerifyShapeInferenceType(outputShape, m_ShapeInferenceMethod);

    auto inferredShapes = InferOutputShapes({ GetInputSlot(0).GetConnection()->GetTensorInfo().GetShape() });
    ARMNN_ASSERT(inferredShapes.size() == 1);

    ValidateAndCopyShape(outputShape, inferredShapes[0], m_ShapeInferenceMethod, "LogSoftmaxLayer");
}

ARMNN_NO_DEPRECATE_WARN_BEGIN
void LogSoftmaxLayer::Accept(ILayerVisitor& visitor) const
{
    visitor.VisitLogSoftmaxLayer(this, GetParameters(), GetName());
}
ARMNN_NO_DEPRECATE_WARN_END

} // namespace armnn
