//
// Copyright © 2017 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "SpaceToBatchNdLayer.hpp"
#include "LayerCloneBase.hpp"

#include <armnn/TypesUtils.hpp>

#include <armnnUtils/DataLayoutIndexed.hpp>

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
    ARMNN_ASSERT(inputShapes.size() == 1);

    TensorShape inputShape = inputShapes[0];
    TensorShape outputShape(inputShape);

    outputShape[0] = inputShape[0] * std::accumulate(m_Param.m_BlockShape.begin(),
                                                     m_Param.m_BlockShape.end(),
                                                     1U,
                                                     std::multiplies<>());

    DataLayoutIndexed dimensionIndices = m_Param.m_DataLayout;
    unsigned int heightIndex = dimensionIndices.GetHeightIndex();
    unsigned int widthIndex = dimensionIndices.GetWidthIndex();

    std::pair<unsigned int, unsigned int> heightPad = m_Param.m_PadList[0];
    std::pair<unsigned int, unsigned int> widthPad = m_Param.m_PadList[1];

    outputShape[heightIndex] =
        (inputShape[heightIndex] + heightPad.first + heightPad.second) / m_Param.m_BlockShape[0];
    outputShape[widthIndex] =
        (inputShape[widthIndex] + widthPad.first + widthPad.second) / m_Param.m_BlockShape[1];

    return std::vector<TensorShape>({ outputShape });
}

void SpaceToBatchNdLayer::ValidateTensorShapesFromInputs()
{
    VerifyLayerConnections(1, CHECK_LOCATION());

    const TensorShape& outputShape = GetOutputSlot(0).GetTensorInfo().GetShape();

    VerifyShapeInferenceType(outputShape, m_ShapeInferenceMethod);

    std::vector<TensorShape> inferredShapes = InferOutputShapes({
        GetInputSlot(0).GetConnection()->GetTensorInfo().GetShape() });

    ARMNN_ASSERT(inferredShapes.size() == 1);

    ValidateAndCopyShape(outputShape, inferredShapes[0], m_ShapeInferenceMethod, "SpaceToBatchNdLayer");
}

void SpaceToBatchNdLayer::ExecuteStrategy(IStrategy& strategy) const
{
    strategy.ExecuteStrategy(this, GetParameters(), {}, GetName());
}

} // namespace
