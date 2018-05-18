//
// Copyright © 2017 Arm Ltd. All rights reserved.
// See LICENSE file in the project root for full license information.
//
#include "ReshapeLayer.hpp"

#include "LayerCloneBase.hpp"

#include <armnn/TypesUtils.hpp>
#include <backends/WorkloadData.hpp>
#include <backends/WorkloadFactory.hpp>

namespace armnn
{

ReshapeLayer::ReshapeLayer(const ReshapeDescriptor& param, const char* name)
    : LayerWithParameters(1, 1, LayerType::Reshape, param, name)
{
}

std::unique_ptr<IWorkload> ReshapeLayer::CreateWorkload(const Graph& graph,
    const IWorkloadFactory& factory) const
{
    ReshapeQueueDescriptor descriptor;
    return factory.CreateReshape(descriptor, PrepInfoAndDesc(descriptor, graph));
}

ReshapeLayer* ReshapeLayer::Clone(Graph& graph) const
{
    return CloneBase<ReshapeLayer>(graph, m_Param, GetName());
}

void ReshapeLayer::ValidateTensorShapesFromInputs()
{
    ConditionalThrow<LayerValidationException>(GetInputSlot(0).GetConnection() != nullptr,
                     "ReshapeLayer: InputSlot must be connected to an OutputSlot");
    ConditionalThrow<LayerValidationException>(GetInputSlot(0).GetConnection()->IsTensorInfoSet(),
                     "ReshapeLayer: TensorInfo must be set on connected OutputSlot.");

    ConditionalThrowIfNotEqual<LayerValidationException>(
        "ReshapeLayer: TensorShape set on OutputSlot[0] does not match the inferred shape.",
        GetOutputSlot(0).GetTensorInfo().GetShape(),
        m_Param.m_TargetShape);
}

} // namespace armnn
