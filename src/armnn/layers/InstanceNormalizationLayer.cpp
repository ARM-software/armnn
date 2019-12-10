//
// Copyright © 2019 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//
#include "InstanceNormalizationLayer.hpp"

#include "LayerCloneBase.hpp"

#include <armnn/TypesUtils.hpp>
#include <backendsCommon/WorkloadData.hpp>
#include <backendsCommon/WorkloadFactory.hpp>

namespace armnn
{

InstanceNormalizationLayer::InstanceNormalizationLayer(const InstanceNormalizationDescriptor& param, const char* name)
    : LayerWithParameters(1, 1, LayerType::InstanceNormalization, param, name)
{
}

std::unique_ptr<IWorkload> InstanceNormalizationLayer::CreateWorkload(const IWorkloadFactory& factory) const
{
    InstanceNormalizationQueueDescriptor descriptor;
    return factory.CreateInstanceNormalization(descriptor, PrepInfoAndDesc(descriptor));
}

InstanceNormalizationLayer* InstanceNormalizationLayer::Clone(Graph& graph) const
{
    return CloneBase<InstanceNormalizationLayer>(graph, m_Param, GetName());
}

void InstanceNormalizationLayer::ValidateTensorShapesFromInputs()
{
    VerifyLayerConnections(1, CHECK_LOCATION());

    auto inferredShapes = InferOutputShapes({ GetInputSlot(0).GetConnection()->GetTensorInfo().GetShape() });

    BOOST_ASSERT(inferredShapes.size() == 1);

    ConditionalThrowIfNotEqual<LayerValidationException>(
        "InstanceNormalizationLayer: TensorShape set on OutputSlot[0] does not match the inferred shape.",
        GetOutputSlot(0).GetTensorInfo().GetShape(),
        inferredShapes[0]);
}

void InstanceNormalizationLayer::Accept(ILayerVisitor& visitor) const
{
    visitor.VisitInstanceNormalizationLayer(this, GetParameters(), GetName());
}

} // namespace armnn
