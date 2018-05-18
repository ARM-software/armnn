//
// Copyright © 2017 Arm Ltd. All rights reserved.
// See LICENSE file in the project root for full license information.
//
#include "PermuteLayer.hpp"

#include "LayerCloneBase.hpp"

#include <armnn/TypesUtils.hpp>
#include <backends/WorkloadData.hpp>
#include <backends/WorkloadFactory.hpp>

#include <Permute.hpp>

namespace armnn
{

PermuteLayer::PermuteLayer(const PermuteDescriptor& param, const char* name)
    : LayerWithParameters(1, 1, LayerType::Permute, param, name)
{
}

std::unique_ptr<IWorkload> PermuteLayer::CreateWorkload(const Graph& graph, const IWorkloadFactory& factory) const
{
    PermuteQueueDescriptor descriptor;
    return factory.CreatePermute(descriptor, PrepInfoAndDesc(descriptor, graph));
}

PermuteLayer* PermuteLayer::Clone(Graph& graph) const
{
    return CloneBase<PermuteLayer>(graph, m_Param, GetName());
}

void PermuteLayer::ValidateTensorShapesFromInputs()
{
    ConditionalThrow<LayerValidationException>(GetInputSlot(0).GetConnection() != nullptr,
                     "PermuteLayer: InputSlot must be connected to an OutputSlot");
    ConditionalThrow<LayerValidationException>(GetInputSlot(0).GetConnection()->IsTensorInfoSet(),
                     "PermuteLayer: TensorInfo must be set on connected InputSlot.");

    const TensorInfo& infoIn = GetInputSlot(0).GetConnection()->GetTensorInfo();
    TensorShape shapeOut = armnnUtils::Permuted(infoIn.GetShape(), m_Param.m_DimMappings);
    ConditionalThrowIfNotEqual<LayerValidationException>(
        "PermuteLayer: TensorShape set on OutputSlot[0] does not match the inferred shape.",
        GetOutputSlot(0).GetTensorInfo().GetShape(),
        shapeOut);
}

} // namespace armnn
