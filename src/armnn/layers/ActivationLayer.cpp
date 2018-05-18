//
// Copyright © 2017 Arm Ltd. All rights reserved.
// See LICENSE file in the project root for full license information.
//
#include "ActivationLayer.hpp"
#include "LayerCloneBase.hpp"

#include <armnn/TypesUtils.hpp>
#include <backends/WorkloadData.hpp>
#include <backends/WorkloadFactory.hpp>

namespace armnn
{

ActivationLayer::ActivationLayer(const ActivationDescriptor& param, const char* name)
    : LayerWithParameters(1, 1, LayerType::Activation, param, name)
{
}

std::unique_ptr<IWorkload> ActivationLayer::CreateWorkload(const Graph& graph, const IWorkloadFactory& factory) const
{
    ActivationQueueDescriptor descriptor;
    return factory.CreateActivation(descriptor, PrepInfoAndDesc(descriptor, graph));
}

ActivationLayer* ActivationLayer::Clone(Graph& graph) const
{
    return CloneBase<ActivationLayer>(graph, m_Param, GetName());
}

void ActivationLayer::ValidateTensorShapesFromInputs()
{
    auto& info = GetInputSlot(0).GetConnection()->GetTensorInfo();

    ConditionalThrowIfNotEqual<LayerValidationException>(
        "ActivationLayer: TensorShape set on OutputSlot[0] does not match the inferred shape.",
        GetOutputSlot(0).GetTensorInfo().GetShape(),
        info.GetShape());
}

} // namespace armnn
