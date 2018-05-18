//
// Copyright © 2017 Arm Ltd. All rights reserved.
// See LICENSE file in the project root for full license information.
//
#include "BatchNormalizationLayer.hpp"

#include "LayerCloneBase.hpp"

#include <armnn/TypesUtils.hpp>
#include <backends/CpuTensorHandle.hpp>
#include <backends/WorkloadFactory.hpp>

namespace armnn
{

BatchNormalizationLayer::BatchNormalizationLayer(const armnn::BatchNormalizationDescriptor& param, const char* name)
    : LayerWithParameters(1, 1, LayerType::BatchNormalization, param, name)
{
}

std::unique_ptr<IWorkload> BatchNormalizationLayer::CreateWorkload(const Graph& graph,
                                                                   const IWorkloadFactory& factory) const
{
    BatchNormalizationQueueDescriptor descriptor;

    descriptor.m_Mean = m_Mean.get();
    descriptor.m_Variance = m_Variance.get();
    descriptor.m_Beta = m_Beta.get();
    descriptor.m_Gamma = m_Gamma.get();
    return factory.CreateBatchNormalization(descriptor, PrepInfoAndDesc(descriptor, graph));
}

BatchNormalizationLayer* BatchNormalizationLayer::Clone(Graph& graph) const
{
    auto layer = CloneBase<BatchNormalizationLayer>(graph, m_Param, GetName());

    layer->m_Mean = m_Mean ? std::make_unique<ScopedCpuTensorHandle>(*m_Mean) : nullptr;
    layer->m_Variance = m_Variance ? std::make_unique<ScopedCpuTensorHandle>(*m_Variance) : nullptr;
    layer->m_Beta = m_Beta ? std::make_unique<ScopedCpuTensorHandle>(*m_Beta) : nullptr;
    layer->m_Gamma = m_Gamma ? std::make_unique<ScopedCpuTensorHandle>(*m_Gamma) : nullptr;

    return std::move(layer);
}

void BatchNormalizationLayer::ValidateTensorShapesFromInputs()
{
    ConditionalThrow<LayerValidationException>(GetInputSlot(0).GetConnection() != nullptr,
                     "BatchNormalizationLayer: InputSlot must be connected to an OutputSlot");
    ConditionalThrow<LayerValidationException>(GetInputSlot(0).GetConnection()->IsTensorInfoSet(),
                     "BatchNormalizationLayer: TensorInfo must be set on connected OutputSlot.");

    auto& info = GetInputSlot(0).GetConnection()->GetTensorInfo();

    ConditionalThrowIfNotEqual<LayerValidationException>(
        "BatchNormalizationLayer: TensorShape set on OutputSlot[0] does not match the inferred shape.",
        GetOutputSlot(0).GetTensorInfo().GetShape(),
        info.GetShape());
}

} // namespace armnn
