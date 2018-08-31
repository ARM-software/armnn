//
// Copyright © 2017 Arm Ltd. All rights reserved.
// See LICENSE file in the project root for full license information.
//
#include "ConvertFp32ToFp16Layer.hpp"

#include "LayerCloneBase.hpp"

#include <armnn/TypesUtils.hpp>
#include <backends/WorkloadData.hpp>
#include <backends/WorkloadFactory.hpp>

namespace armnn
{

ConvertFp32ToFp16Layer::ConvertFp32ToFp16Layer(const char* name)
 : Layer(1, 1, LayerType::ConvertFp32ToFp16, name)
{
}

std::unique_ptr<IWorkload> ConvertFp32ToFp16Layer::CreateWorkload(const Graph& graph,
    const IWorkloadFactory& factory) const
{
    ConvertFp32ToFp16QueueDescriptor descriptor;
    return factory.CreateConvertFp32ToFp16(descriptor, PrepInfoAndDesc(descriptor, graph));
}

ConvertFp32ToFp16Layer* ConvertFp32ToFp16Layer::Clone(Graph& graph) const
{
    return CloneBase<ConvertFp32ToFp16Layer>(graph, GetName());
}

void ConvertFp32ToFp16Layer::ValidateTensorShapesFromInputs()
{
    VerifyLayerConnections(1, CHECK_LOCATION());

    auto inferredShapes = InferOutputShapes({ GetInputSlot(0).GetConnection()->GetTensorInfo().GetShape() });

    BOOST_ASSERT(inferredShapes.size() == 1);

    ConditionalThrowIfNotEqual<LayerValidationException>(
        "ConvertFp32ToFp16Layer: TensorShape set on OutputSlot[0] does not match the inferred shape.",
        GetOutputSlot(0).GetTensorInfo().GetShape(),
        inferredShapes[0]);
}

} // namespace armnn
