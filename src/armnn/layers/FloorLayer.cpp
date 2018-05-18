//
// Copyright © 2017 Arm Ltd. All rights reserved.
// See LICENSE file in the project root for full license information.
//
#include "FloorLayer.hpp"

#include "LayerCloneBase.hpp"

#include <armnn/TypesUtils.hpp>
#include <backends/WorkloadData.hpp>
#include <backends/WorkloadFactory.hpp>

namespace armnn
{

FloorLayer::FloorLayer(const char* name)
 : Layer(1, 1, LayerType::Floor, name)
{
}

std::unique_ptr<IWorkload> FloorLayer::CreateWorkload(const Graph& graph,
    const IWorkloadFactory& factory) const
{
    FloorQueueDescriptor descriptor;
    return factory.CreateFloor(descriptor, PrepInfoAndDesc(descriptor, graph));
}

FloorLayer* FloorLayer::Clone(Graph& graph) const
{
    return CloneBase<FloorLayer>(graph, GetName());
}

void FloorLayer::ValidateTensorShapesFromInputs()
{
    ConditionalThrow<LayerValidationException>(GetInputSlot(0).GetConnection() != nullptr,
                     "FloorLayer: InputSlot must be connected to an OutputSlot");
    ConditionalThrow<LayerValidationException>(GetInputSlot(0).GetConnection()->IsTensorInfoSet(),
                     "FloorLayer: TensorInfo must be set on connected OutputSlot.");

    // input and output shapes are the same
    IOutputSlot* input = GetInputSlot(0).GetConnection();
    TensorShape const& outShape = input->GetTensorInfo().GetShape();
    ConditionalThrowIfNotEqual<LayerValidationException>(
        "FloorLayer: TensorShape set on OutputSlot[0] does not match the inferred shape.",
        GetOutputSlot(0).GetTensorInfo().GetShape(),
        outShape);
}

} // namespace armnn
