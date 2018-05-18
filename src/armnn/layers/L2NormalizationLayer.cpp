//
// Copyright © 2017 Arm Ltd. All rights reserved.
// See LICENSE file in the project root for full license information.
//
#include "L2NormalizationLayer.hpp"

#include "LayerCloneBase.hpp"

#include <armnn/TypesUtils.hpp>
#include <backends/WorkloadData.hpp>
#include <backends/WorkloadFactory.hpp>

namespace armnn
{

L2NormalizationLayer::L2NormalizationLayer(const char* name)
    : Layer(1, 1, LayerType::L2Normalization, name)
{
}

std::unique_ptr<IWorkload> L2NormalizationLayer::CreateWorkload(const Graph& graph,
    const IWorkloadFactory& factory) const
{
    L2NormalizationQueueDescriptor descriptor;
    return factory.CreateL2Normalization(descriptor, PrepInfoAndDesc(descriptor, graph));
}

L2NormalizationLayer* L2NormalizationLayer::Clone(Graph& graph) const
{
    return CloneBase<L2NormalizationLayer>(graph, GetName());
}

void L2NormalizationLayer::ValidateTensorShapesFromInputs()
{
    ConditionalThrow<LayerValidationException>(GetInputSlot(0).GetConnection() != nullptr,
                     "L2NormalizationLayer: InputSlot must be connected to an OutputSlot");
    ConditionalThrow<LayerValidationException>(GetInputSlot(0).GetConnection()->IsTensorInfoSet(),
                     "L2NormalizationLayer: TensorInfo must be set on connected OutputSlot.");

    IOutputSlot* input = GetInputSlot(0).GetConnection();

    // input and output shapes are the same
    TensorShape const& outShape = input->GetTensorInfo().GetShape();
    ConditionalThrowIfNotEqual<LayerValidationException>(
        "L2NormalizationLayer: TensorShape set on OutputSlot[0] does not match the inferred shape.",
        GetOutputSlot(0).GetTensorInfo().GetShape(),
        outShape);
}

} // namespace armnn
