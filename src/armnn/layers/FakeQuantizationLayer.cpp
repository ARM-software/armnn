//
// Copyright © 2017 Arm Ltd. All rights reserved.
// See LICENSE file in the project root for full license information.
//
#include "FakeQuantizationLayer.hpp"

#include "LayerCloneBase.hpp"

#include <armnn/TypesUtils.hpp>
#include <backends/WorkloadData.hpp>
#include <backends/WorkloadFactory.hpp>

namespace armnn
{

FakeQuantizationLayer::FakeQuantizationLayer(const FakeQuantizationDescriptor& param, const char* name)
: LayerWithParameters(1, 1, LayerType::FakeQuantization, param, name)
{
}

std::unique_ptr<IWorkload> FakeQuantizationLayer::CreateWorkload(const Graph& graph,
                                                                const IWorkloadFactory& factory) const
{
    FakeQuantizationQueueDescriptor descriptor;
    return factory.CreateFakeQuantization(descriptor, PrepInfoAndDesc(descriptor, graph) );
}

FakeQuantizationLayer* FakeQuantizationLayer::Clone(Graph& graph) const
{
    return CloneBase<FakeQuantizationLayer>(graph, m_Param, GetName());
}

void FakeQuantizationLayer::ValidateTensorShapesFromInputs()
{
    ConditionalThrow<LayerValidationException>(GetInputSlot(0).GetConnection() != nullptr,
                     "FakeQuantizationLayer: InputSlot must be connected to an OutputSlot");
    ConditionalThrow<LayerValidationException>(GetInputSlot(0).GetConnection()->IsTensorInfoSet(),
                     "FakeQuantizationLayer: TensorInfo must be set on connected OutputSlot.");


    IOutputSlot* input = GetInputSlot(0).GetConnection();

    // input and output shapes are the same
    TensorShape const& outShape = input->GetTensorInfo().GetShape();
    ConditionalThrowIfNotEqual<LayerValidationException>(
        "FakeQuantizationLayer: TensorShape set on OutputSlot[0] does not match the inferred shape.",
        GetOutputSlot(0).GetTensorInfo().GetShape(),
        outShape);
}

} // namespace armnn
