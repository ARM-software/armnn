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
    // on this level constant data should not be released..
    BOOST_ASSERT_MSG(m_Weight != nullptr, "FullyConnectedLayer: Weights data should not be null.");

    FullyConnectedQueueDescriptor descriptor;

    descriptor.m_Weight = m_Weight.get();
    if (m_Param.m_BiasEnabled)
    {
        BOOST_ASSERT_MSG(m_Bias != nullptr, "FullyConnectedLayer: Bias data should not be null.");
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

std::vector<TensorShape> FullyConnectedLayer::InferOutputShapes(const std::vector<TensorShape>& inputShapes) const
{
    BOOST_ASSERT(inputShapes.size() == 2);
    const TensorShape& inputShape = inputShapes[0];
    const TensorShape weightShape = inputShapes[1];

    // Output for FC is [1, w[1]].
    unsigned int batches = inputShape[0];
    unsigned int dimIdx = m_Param.m_TransposeWeightMatrix ? 0 : 1;

    return std::vector<TensorShape>({ TensorShape({batches, weightShape[dimIdx]})});
}

void FullyConnectedLayer::ValidateTensorShapesFromInputs()
{
    VerifyLayerConnections(1, CHECK_LOCATION());

    // check if we m_Weight data is not nullptr
    BOOST_ASSERT_MSG(m_Weight != nullptr, "FullyConnectedLayer: Weights data should not be null.");

    auto inferredShapes = InferOutputShapes({
        GetInputSlot(0).GetConnection()->GetTensorInfo().GetShape(),
        m_Weight->GetTensorInfo().GetShape() });

    BOOST_ASSERT(inferredShapes.size() == 1);

    ConditionalThrowIfNotEqual<LayerValidationException>(
        "FullyConnectedLayer: TensorShape set on OutputSlot[0] does not match the inferred shape.",
        GetOutputSlot(0).GetTensorInfo().GetShape(),
        inferredShapes[0]);
}

Layer::ConstantTensors FullyConnectedLayer::GetConstantTensorsByRef()
{
    return {m_Weight, m_Bias};
}

} // namespace armnn
