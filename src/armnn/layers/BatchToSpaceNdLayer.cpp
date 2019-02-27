//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//
#include "BatchToSpaceNdLayer.hpp"

#include "LayerCloneBase.hpp"
#include "LayerWithParameters.hpp"
#include "BatchToSpaceNdLayer.hpp"

#include <armnn/TypesUtils.hpp>

#include <backendsCommon/CpuTensorHandle.hpp>
#include <backendsCommon/WorkloadData.hpp>
#include <backendsCommon/WorkloadFactory.hpp>

#include <DataLayoutIndexed.hpp>

#include <numeric>

using namespace armnnUtils;

namespace armnn
{

BatchToSpaceNdLayer::BatchToSpaceNdLayer(const armnn::BatchToSpaceNdDescriptor& param, const char* name)
    : LayerWithParameters(1, 1, LayerType::BatchToSpaceNd, param, name)
{
}

std::unique_ptr<IWorkload> BatchToSpaceNdLayer::CreateWorkload(const Graph& graph,
                                                               const IWorkloadFactory& factory) const
{
    BatchToSpaceNdQueueDescriptor descriptor;

    return factory.CreateBatchToSpaceNd(descriptor, PrepInfoAndDesc(descriptor, graph));
}

BatchToSpaceNdLayer* BatchToSpaceNdLayer::Clone(Graph& graph) const
{
    auto layer = CloneBase<BatchToSpaceNdLayer>(graph, m_Param, GetName());
    return std::move(layer);
}

void BatchToSpaceNdLayer::ValidateTensorShapesFromInputs()
{
    VerifyLayerConnections(1, CHECK_LOCATION());

    auto inferredShapes = InferOutputShapes({GetInputSlot(0).GetConnection()->GetTensorInfo().GetShape()});

    BOOST_ASSERT(inferredShapes.size() == 1);

    ConditionalThrowIfNotEqual<LayerValidationException>(
        "BatchToSpaceLayer: TensorShape set on OutputSlot[0] does not match the inferred shape.",
        GetOutputSlot(0).GetTensorInfo().GetShape(),inferredShapes[0]);
}

std::vector<TensorShape> BatchToSpaceNdLayer::InferOutputShapes(const std::vector<TensorShape>& inputShapes) const
{
    BOOST_ASSERT(inputShapes.size() == 1);

    const TensorShape& inputShape = inputShapes[0];
    TensorShape outputShape(inputShape);

    unsigned int accumulatedBlockShape = std::accumulate(m_Param.m_BlockShape.begin(),
                                                         m_Param.m_BlockShape.end(),
                                                         1U,
                                                         std::multiplies<>());

    BOOST_ASSERT(inputShape[0] % accumulatedBlockShape == 0);

    outputShape[0] = inputShape[0] / accumulatedBlockShape;

    DataLayoutIndexed dimensionIndices = m_Param.m_DataLayout;
    unsigned int heightIndex = dimensionIndices.GetHeightIndex();
    unsigned int widthIndex = dimensionIndices.GetWidthIndex();

    unsigned int heightCrop = m_Param.m_Crops[0].first + m_Param.m_Crops[0].second;
    unsigned int widthCrop = m_Param.m_Crops[1].first + m_Param.m_Crops[1].second;

    unsigned int outputHeight = inputShape[heightIndex] * m_Param.m_BlockShape[0];
    unsigned int outputWidth = inputShape[widthIndex] * m_Param.m_BlockShape[1];

    BOOST_ASSERT_MSG(heightCrop <= outputHeight,
        "BatchToSpaceLayer: Overall height crop should be less than or equal to the uncropped output height.");

    BOOST_ASSERT_MSG(widthCrop <= outputWidth,
        "BatchToSpaceLayer: Overall width crop should be less than or equal to the uncropped output width.");

    outputShape[heightIndex] = outputHeight - heightCrop;
    outputShape[widthIndex] = outputWidth - widthCrop;

    return std::vector<TensorShape>({ outputShape });
}

void BatchToSpaceNdLayer::Accept(ILayerVisitor& visitor) const
{
    visitor.VisitBatchToSpaceNdLayer(this, GetParameters(), GetName());
}

} // namespace armnn
