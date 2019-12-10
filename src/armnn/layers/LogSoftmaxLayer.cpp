//
// Copyright © 2019 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "LogSoftmaxLayer.hpp"

#include "LayerCloneBase.hpp"

#include <armnn/TypesUtils.hpp>

#include <backendsCommon/WorkloadData.hpp>
#include <backendsCommon/WorkloadFactory.hpp>

namespace armnn
{

LogSoftmaxLayer::LogSoftmaxLayer(const LogSoftmaxDescriptor &param, const char* name)
    : LayerWithParameters(1, 1, LayerType::LogSoftmax, param, name) {}

std::unique_ptr<IWorkload> LogSoftmaxLayer::CreateWorkload(const IWorkloadFactory& factory) const
{
    LogSoftmaxQueueDescriptor descriptor;
    return factory.CreateLogSoftmax(descriptor, PrepInfoAndDesc(descriptor));
}

LogSoftmaxLayer* LogSoftmaxLayer::Clone(Graph& graph) const
{
    return CloneBase<LogSoftmaxLayer>(graph, m_Param, GetName());
}

void LogSoftmaxLayer::ValidateTensorShapesFromInputs()
{
    VerifyLayerConnections(1, CHECK_LOCATION());

    auto inferredShapes = InferOutputShapes({ GetInputSlot(0).GetConnection()->GetTensorInfo().GetShape() });
    BOOST_ASSERT(inferredShapes.size() == 1);

    ConditionalThrowIfNotEqual<LayerValidationException>(
        "LogSoftmaxLayer: TensorShape set on OutputSlot[0] does not match the inferred shape.",
        GetOutputSlot(0).GetTensorInfo().GetShape(),
        inferredShapes[0]);
}

void LogSoftmaxLayer::Accept(ILayerVisitor& visitor) const
{
    visitor.VisitLogSoftmaxLayer(this, GetParameters(), GetName());
}

} // namespace armnn
