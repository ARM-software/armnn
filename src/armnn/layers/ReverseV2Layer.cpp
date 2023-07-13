//
// Copyright © 2023 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "ReverseV2Layer.hpp"

#include <armnn/backends/WorkloadFactory.hpp>
#include "layers/LayerCloneBase.hpp"

namespace armnn
{
ReverseV2Layer::ReverseV2Layer(const char* name)
    : Layer(2, 1, LayerType::ReverseV2, name)
{
}

std::unique_ptr<IWorkload> ReverseV2Layer::CreateWorkload(const armnn::IWorkloadFactory &factory) const
{
    ReverseV2QueueDescriptor descriptor;
    SetAdditionalInfo(descriptor);

    return factory.CreateWorkload(LayerType::ReverseV2, descriptor, PrepInfoAndDesc(descriptor));
}

ReverseV2Layer* ReverseV2Layer::Clone(armnn::Graph &graph) const
{
    auto layer = CloneBase<ReverseV2Layer>(graph, GetName());

    return std::move(layer);
}

std::vector<TensorShape> ReverseV2Layer::InferOutputShapes(const std::vector<TensorShape>& inputShapes) const
{
    ARMNN_ASSERT(inputShapes.size() == 2);

    const auto inputDims = inputShapes[0].GetNumDimensions();

    std::vector<unsigned int> dimSizes(inputDims);
    for (unsigned i=0;  i<inputDims;  i++)
    {
        dimSizes[i] = inputShapes[0][i];
    }

    TensorShape outputShape({ inputDims, dimSizes.data() });

    return std::vector<TensorShape>({ outputShape });
}

void ReverseV2Layer::ValidateTensorShapesFromInputs()
{
    VerifyLayerConnections(2, CHECK_LOCATION());

    const TensorShape& outputShape = GetOutputSlot(0).GetTensorInfo().GetShape();

    VerifyShapeInferenceType(outputShape, m_ShapeInferenceMethod);

    auto inferredShapes = InferOutputShapes({
        GetInputSlot(0).GetTensorInfo().GetShape(),
        GetInputSlot(1).GetTensorInfo().GetShape()});

    ARMNN_ASSERT(inferredShapes.size() == 1);

    ValidateAndCopyShape(outputShape, inferredShapes[0], m_ShapeInferenceMethod, "ReverseV2Layer");
}

void ReverseV2Layer::ExecuteStrategy(IStrategy& strategy) const
{
    strategy.ExecuteStrategy(this, BaseDescriptor(), {}, GetName());
}

}
