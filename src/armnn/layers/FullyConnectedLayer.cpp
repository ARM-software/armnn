//
// Copyright © 2017 Arm Ltd. All rights reserved.
// See LICENSE file in the project root for full license information.
//
#include "FullyConnectedLayer.hpp"

#include "LayerCloneBase.hpp"

#include <armnn/TypesUtils.hpp>
#include <backends/CpuTensorHandle.hpp>
#include <backends/WorkloadData.hpp>
#include <backends/WorkloadFactory.hpp>

namespace armnn
{

FullyConnectedLayer::FullyConnectedLayer(const FullyConnectedDescriptor& param, const char* name)
    : LayerWithParameters(1, 1, LayerType::FullyConnected, param, name)
{
}

std::unique_ptr<IWorkload> FullyConnectedLayer::CreateWorkload(const Graph& graph,
                                                               const IWorkloadFactory& factory) const
{
    FullyConnectedQueueDescriptor descriptor;

    descriptor.m_Weight = m_Weight.get();
    if (m_Param.m_BiasEnabled)
    {
        descriptor.m_Bias = m_Bias.get();
    }
    return factory.CreateFullyConnected(descriptor, PrepInfoAndDesc(descriptor, graph));
}

FullyConnectedLayer* FullyConnectedLayer::Clone(Graph& graph) const
{
    auto layer = CloneBase<FullyConnectedLayer>(graph, m_Param, GetName());

    layer->m_Weight = m_Weight ? std::make_unique<ScopedCpuTensorHandle>(*m_Weight) : nullptr;
    if (layer->m_Param.m_BiasEnabled)
    {
        layer->m_Bias = m_Bias ? std::make_unique<ScopedCpuTensorHandle>(*m_Bias) : nullptr;
    }

    return std::move(layer);
}

void FullyConnectedLayer::ValidateTensorShapesFromInputs()
{
    ConditionalThrow<LayerValidationException>(GetInputSlot(0).GetConnection() != nullptr,
                     "FullyConnectedLayer: InputSlot must be connected to an OutputSlot");
    ConditionalThrow<LayerValidationException>(GetInputSlot(0).GetConnection()->IsTensorInfoSet(),
                     "FullyConnectedLayer: TensorInfo must be set on connected OutputSlot.");


    TensorShape const& weightShape = m_Weight->GetTensorInfo().GetShape();

    // output for FC is [1, w[1]]
    unsigned int batches = GetInputSlot(0).GetConnection()->GetTensorInfo().GetShape()[0];
    unsigned int dimIdx = m_Param.m_TransposeWeightMatrix ? 0 : 1;
    TensorShape outShape({batches, weightShape[dimIdx]});

    ConditionalThrowIfNotEqual<LayerValidationException>(
        "FullyConnectedLayer: TensorShape set on OutputSlot[0] does not match the inferred shape.",
        GetOutputSlot(0).GetTensorInfo().GetShape(),
        outShape);
}

} // namespace armnn
