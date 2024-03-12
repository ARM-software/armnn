//
// Copyright © 2018-2024 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "SpaceToBatchNdLayer.hpp"
#include "LayerCloneBase.hpp"

#include <armnn/backends/WorkloadData.hpp>
#include <armnn/backends/WorkloadFactory.hpp>

#include <numeric>

using namespace armnnUtils;

namespace armnn
{

SpaceToBatchNdLayer::SpaceToBatchNdLayer(const SpaceToBatchNdDescriptor param, const char* name)
    : LayerWithParameters(1, 1, LayerType::SpaceToBatchNd, param, name)
{}

std::unique_ptr<IWorkload> SpaceToBatchNdLayer::CreateWorkload(const IWorkloadFactory& factory) const
{
    SpaceToBatchNdQueueDescriptor descriptor;
    descriptor.m_Parameters.m_BlockShape = m_Param.m_BlockShape;
    descriptor.m_Parameters.m_PadList    = m_Param.m_PadList;
    SetAdditionalInfo(descriptor);

    return factory.CreateWorkload(LayerType::SpaceToBatchNd, descriptor, PrepInfoAndDesc(descriptor));
}

SpaceToBatchNdLayer* SpaceToBatchNdLayer::Clone(Graph& graph) const
{
    IgnoreUnused(graph);
    return CloneBase<SpaceToBatchNdLayer>(graph, m_Param, GetName());
}

std::vector<TensorShape> SpaceToBatchNdLayer::InferOutputShapes(const std::vector<TensorShape>& inputShapes) const
{
    const TensorShape inputShape = inputShapes[0];
    TensorShape outputShape(inputShape);

    outputShape[0] = inputShape[0] * std::accumulate(m_Param.m_BlockShape.begin(),
                                                     m_Param.m_BlockShape.end(),
                                                     1U,
                                                     std::multiplies<>());

    // In a 4D tensor, there will be 2 spatialDimensions (H and W), and the for loop will run twice.
    // In a 3D tensor, there will be 1 spatialDimensions, and the for loop will run once.
    unsigned int firstSpatialDimension = m_Param.m_DataLayout == DataLayout::NCHW ? 2 : 1;
    for (unsigned int i = 0; i < m_Param.m_BlockShape.size(); ++i)
    {
        unsigned int spatialDimension = firstSpatialDimension + i;
        outputShape[spatialDimension] = 
          (inputShape[spatialDimension] + m_Param.m_PadList[i].first + m_Param.m_PadList[i].second)
          / m_Param.m_BlockShape[i];
    }

    return std::vector<TensorShape>({ outputShape });
}

void SpaceToBatchNdLayer::ValidateTensorShapesFromInputs()
{
    VerifyLayerConnections(1, CHECK_LOCATION());

    const TensorShape& outputShape = GetOutputSlot(0).GetTensorInfo().GetShape();

    VerifyShapeInferenceType(outputShape, m_ShapeInferenceMethod);

    std::vector<TensorShape> inferredShapes = InferOutputShapes({
        GetInputSlot(0).GetTensorInfo().GetShape() });

    if (inferredShapes.size() != 1)
    {
        throw armnn::LayerValidationException("inferredShapes has "
                                              + std::to_string(inferredShapes.size()) +
                                              " elements - should only have 1.");
    }

    ValidateAndCopyShape(outputShape, inferredShapes[0], m_ShapeInferenceMethod, "SpaceToBatchNdLayer");
}

void SpaceToBatchNdLayer::ExecuteStrategy(IStrategy& strategy) const
{
    strategy.ExecuteStrategy(this, GetParameters(), {}, GetName());
}

} // namespace
