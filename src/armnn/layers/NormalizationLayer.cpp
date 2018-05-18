//
// Copyright © 2017 Arm Ltd. All rights reserved.
// See LICENSE file in the project root for full license information.
//
#include "NormalizationLayer.hpp"

#include "LayerCloneBase.hpp"

#include <armnn/TypesUtils.hpp>
#include <backends/WorkloadData.hpp>
#include <backends/WorkloadFactory.hpp>

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
    ConditionalThrow<LayerValidationException>(GetInputSlot(0).GetConnection() != nullptr,
                                               "NormalizationLayer: Input slot must be connected.");

    const TensorShape& outShape = GetInputSlot(0).GetConnection()->GetTensorInfo().GetShape();
    ConditionalThrowIfNotEqual<LayerValidationException>(
        "NormalizationLayer: TensorShape set on OutputSlot[0] does not match the inferred shape.",
        GetOutputSlot(0).GetTensorInfo().GetShape(),
        outShape);
}

} // namespace armnn
