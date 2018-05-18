//
// Copyright © 2017 Arm Ltd. All rights reserved.
// See LICENSE file in the project root for full license information.
//
#include "MemCopyLayer.hpp"

#include "LayerCloneBase.hpp"

#include <armnn/TypesUtils.hpp>
#include <backends/WorkloadData.hpp>
#include <backends/WorkloadFactory.hpp>

namespace armnn
{

MemCopyLayer::MemCopyLayer(const char* name)
    : Layer(1, 1, LayerType::MemCopy, name)
{
}

MemCopyLayer* MemCopyLayer::Clone(Graph& graph) const
{
    return CloneBase<MemCopyLayer>(graph, GetName());
}

std::unique_ptr<IWorkload> MemCopyLayer::CreateWorkload(const Graph& graph, const IWorkloadFactory& factory) const
{
    MemCopyQueueDescriptor descriptor;
    return factory.CreateMemCopy(descriptor, PrepInfoAndDesc(descriptor, graph));
}

void MemCopyLayer::ValidateTensorShapesFromInputs()
{
    ConditionalThrow<LayerValidationException>(GetInputSlot(0).GetConnection() != nullptr,
                     "MemCopyLayer: InputSlot must be connected to an OutputSlot");
    ConditionalThrow<LayerValidationException>(GetInputSlot(0).GetConnection()->IsTensorInfoSet(),
                     "MemCopyLayer: TensorInfo must be set on connected OutputSlot.");


    IOutputSlot* input = GetInputSlot(0).GetConnection();

    ConditionalThrowIfNotEqual<LayerValidationException>(
        "MemCopyLayer: TensorShape set on OutputSlot[0] does not match the inferred shape.",
        GetOutputSlot(0).GetTensorInfo().GetShape(),
        input->GetTensorInfo().GetShape());
}

} // namespace armnn
