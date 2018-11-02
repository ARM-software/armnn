//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "SpaceToBatchNdLayer.hpp"

#include "LayerCloneBase.hpp"

#include <armnn/TypesUtils.hpp>
#include <backends/WorkloadData.hpp>
#include <backends/WorkloadFactory.hpp>

#include <numeric>

namespace armnn
{

SpaceToBatchNdLayer::SpaceToBatchNdLayer(const SpaceToBatchNdDescriptor param, const char* name)
    : LayerWithParameters(1, 1, LayerType::SpaceToBatchNd, param, name)
{}

std::unique_ptr<IWorkload> SpaceToBatchNdLayer::CreateWorkload(const Graph& graph,
                                                               const IWorkloadFactory& factory) const
{
    SpaceToBatchNdQueueDescriptor descriptor;
    descriptor.m_Parameters.m_BlockShape = m_Param.m_BlockShape;
    descriptor.m_Parameters.m_PadList = m_Param.m_PadList;

    return factory.CreateSpaceToBatchNd(descriptor, PrepInfoAndDesc(descriptor, graph));
}

SpaceToBatchNdLayer* SpaceToBatchNdLayer::Clone(Graph& graph) const
{
    return CloneBase<SpaceToBatchNdLayer>(graph, m_Param, GetName());
}

std::vector<TensorShape> SpaceToBatchNdLayer::InferOutputShapes(const std::vector<TensorShape>& inputShapes) const
{
    BOOST_ASSERT(inputShapes.size() == 1);

    TensorShape inputShape = inputShapes[0];
    TensorShape outputShape(inputShape);

    outputShape[0] = inputShape[0] * std::accumulate(m_Param.m_BlockShape.begin(),
                                                     m_Param.m_BlockShape.end(),
                                                     1U,
                                                     std::multiplies<>());

    unsigned int heightIndex = m_Param.m_DataLayout.GetHeightIndex();
    unsigned int widthIndex = m_Param.m_DataLayout.GetWidthIndex();

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

    std::vector<TensorShape> inferredShapes = InferOutputShapes({
        GetInputSlot(0).GetConnection()->GetTensorInfo().GetShape() });

    BOOST_ASSERT(inferredShapes.size() == 1);

    ConditionalThrowIfNotEqual<LayerValidationException>(
        "SpaceToBatchNdLayer: TensorShape set on OutputSlot[0] does not match the inferred shape.",
        GetOutputSlot(0).GetTensorInfo().GetShape(),
        inferredShapes[0]);
}

} // namespace
