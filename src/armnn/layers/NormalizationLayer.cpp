//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//
#include "NormalizationLayer.hpp"

#include "LayerCloneBase.hpp"

#include <armnn/TypesUtils.hpp>
#include <backendsCommon/WorkloadData.hpp>
#include <backendsCommon/WorkloadFactory.hpp>

namespace armnn
{

NormalizationLayer::NormalizationLayer(const NormalizationDescriptor& param, const char* name)
    : LayerWithParameters(1, 1, LayerType::Normalization, param, name)
{
}

std::unique_ptr<IWorkload> NormalizationLayer::CreateWorkload(const Graph& graph, const IWorkloadFactory& factory) const
{
    NormalizationQueueDescriptor descriptor;
    return factory.CreateNormalization(descriptor, PrepInfoAndDesc(descriptor, graph));
}

NormalizationLayer* NormalizationLayer::Clone(Graph& graph) const
{
    return CloneBase<NormalizationLayer>(graph, m_Param, GetName());
}

void NormalizationLayer::ValidateTensorShapesFromInputs()
{
    VerifyLayerConnections(1, CHECK_LOCATION());

    auto inferredShapes = InferOutputShapes({ GetInputSlot(0).GetConnection()->GetTensorInfo().GetShape() });

    BOOST_ASSERT(inferredShapes.size() == 1);

    ConditionalThrowIfNotEqual<LayerValidationException>(
        "NormalizationLayer: TensorShape set on OutputSlot[0] does not match the inferred shape.",
        GetOutputSlot(0).GetTensorInfo().GetShape(),
        inferredShapes[0]);
}

void NormalizationLayer::Accept(ILayerVisitor& visitor) const
{
    visitor.VisitNormalizationLayer(this, GetParameters(), GetName());
}

} // namespace armnn
