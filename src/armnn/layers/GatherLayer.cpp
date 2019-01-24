//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "GatherLayer.hpp"
#include "LayerCloneBase.hpp"

#include <armnn/TypesUtils.hpp>
#include <backendsCommon/WorkloadData.hpp>
#include <backendsCommon/WorkloadFactory.hpp>

namespace armnn
{

GatherLayer::GatherLayer(const char* name)
    : Layer(2, 1, LayerType::Gather, name)
{
}

std::unique_ptr<IWorkload> GatherLayer::CreateWorkload(const armnn::Graph& graph,
                                                       const armnn::IWorkloadFactory& factory) const
{
    GatherQueueDescriptor descriptor;
    return factory.CreateGather(descriptor, PrepInfoAndDesc(descriptor, graph));
}

GatherLayer* GatherLayer::Clone(Graph& graph) const
{
    return CloneBase<GatherLayer>(graph, GetName());
}

void GatherLayer::ValidateTensorShapesFromInputs()
{
    VerifyLayerConnections(2, CHECK_LOCATION());

    const TensorInfo& params = GetInputSlot(0).GetConnection()->GetTensorInfo();
    const TensorInfo& indices = GetInputSlot(1).GetConnection()->GetTensorInfo();

    const unsigned int paramsDim = params.GetNumDimensions();
    const unsigned int indicesDim = indices.GetNumDimensions();
    const unsigned int outputDim = paramsDim - 1 + indicesDim;

    std::vector<unsigned int> dimSizes;

    for (unsigned int i = 0; i < indicesDim; ++i)
    {
        dimSizes.push_back(indices.GetShape()[i]);
    }
    for (unsigned int i = 1; i < paramsDim; ++i)
    {
        dimSizes.push_back(params.GetShape()[i]);
    }

    const TensorShape& inferredShape = TensorShape(outputDim, dimSizes.data());

    ConditionalThrowIfNotEqual<LayerValidationException>(
        "GatherLayer: TensorShape set on OutputSlot[0] does not match the inferred shape.",
        GetOutputSlot(0).GetTensorInfo().GetShape(),
        inferredShape);
}

void GatherLayer::Accept(ILayerVisitor& visitor) const
{
    visitor.VisitGatherLayer(this, GetName());
}

} // namespace armnn
