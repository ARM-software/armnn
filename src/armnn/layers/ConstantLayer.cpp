//
// Copyright © 2017 Arm Ltd. All rights reserved.
// See LICENSE file in the project root for full license information.
//
#include "ConstantLayer.hpp"
#include "LayerCloneBase.hpp"

#include <armnn/TypesUtils.hpp>
#include <backends/CpuTensorHandle.hpp>
#include <backends/WorkloadData.hpp>
#include <backends/WorkloadFactory.hpp>

namespace armnn
{

ConstantLayer::ConstantLayer(const std::shared_ptr<ScopedCpuTensorHandle>& input, const char* name)
    : Layer(0, 1, LayerType::Constant, name)
    , m_LayerOutput(input)
{
}

std::unique_ptr<IWorkload> ConstantLayer::CreateWorkload(const Graph& graph,
    const IWorkloadFactory& factory) const
{
    ConstantQueueDescriptor descriptor;
    descriptor.m_LayerOutput = m_LayerOutput.get();
    return factory.CreateConstant(descriptor, PrepInfoAndDesc(descriptor, graph));
}

ConstantLayer* ConstantLayer::Clone(Graph& graph) const
{
    // Cloned layers share the same layer output object
    return CloneBase<ConstantLayer>(graph, m_LayerOutput, GetName());
}

void ConstantLayer::ValidateTensorShapesFromInputs()
{
    // get the output shape from the value of the constant layer
    TensorShape const& outShape = m_LayerOutput->GetTensorInfo().GetShape();
    ConditionalThrowIfNotEqual<LayerValidationException>(
        "ConstantLayer: TensorShape set on OutputSlot[0] does not match the inferred shape.",
        GetOutputSlot(0).GetTensorInfo().GetShape(),
        outShape);
}

} // namespace armnn
