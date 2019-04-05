//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//
#include "MergeLayer.hpp"

#include "LayerCloneBase.hpp"

#include <backendsCommon/WorkloadData.hpp>
#include <backendsCommon/WorkloadFactory.hpp>

namespace armnn
{

MergeLayer::MergeLayer(const char* name)
    : Layer(2, 1, LayerType::Merge, name)
{}

std::unique_ptr<IWorkload> MergeLayer::CreateWorkload(const Graph& graph,
                                                      const IWorkloadFactory& factory) const
{
    return nullptr;
}

MergeLayer* MergeLayer::Clone(Graph& graph) const
{
    return CloneBase<MergeLayer>(graph, GetName());
}

void MergeLayer::ValidateTensorShapesFromInputs()
{
    VerifyLayerConnections(2, CHECK_LOCATION());

    std::vector<TensorShape> inferredShapes = InferOutputShapes({
        GetInputSlot(0).GetConnection()->GetTensorInfo().GetShape(),
        GetInputSlot(1).GetConnection()->GetTensorInfo().GetShape(),
    });

    BOOST_ASSERT(inferredShapes.size() == 1);

    ConditionalThrowIfNotEqual<LayerValidationException>(
        "MergeLayer: TensorShape set on OutputSlot[0] does not match the inferred shape.",
        GetOutputSlot(0).GetTensorInfo().GetShape(),
        inferredShapes[0]);
}

std::vector<TensorShape> MergeLayer::InferOutputShapes(const std::vector<TensorShape>& inputShapes) const
{
    BOOST_ASSERT(inputShapes.size() == 2);

    ConditionalThrowIfNotEqual<LayerValidationException>(
        "MergeLayer: TensorShapes set on inputs do not match",
        inputShapes[0],
        inputShapes[1]
    );

    return {inputShapes[0]};
}

void MergeLayer::Accept(ILayerVisitor& visitor) const
{
    visitor.VisitMergeLayer(this, GetName());
}

} // namespace armnn
