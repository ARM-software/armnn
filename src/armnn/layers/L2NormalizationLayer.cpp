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
    VerifyLayerConnections(1, CHECK_LOCATION());

    auto inferredShapes = InferOutputShapes({ GetInputSlot(0).GetConnection()->GetTensorInfo().GetShape() });

    BOOST_ASSERT(inferredShapes.size() == 1);

    ConditionalThrowIfNotEqual<LayerValidationException>(
        "L2NormalizationLayer: TensorShape set on OutputSlot[0] does not match the inferred shape.",
        GetOutputSlot(0).GetTensorInfo().GetShape(),
        inferredShapes[0]);
}

} // namespace armnn
