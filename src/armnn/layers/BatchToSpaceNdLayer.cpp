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
    const DataLayoutIndexed dataLayout = m_Param.m_DataLayout;
    const TensorShape& inputShape = inputShapes[0];
    unsigned int inBatchSize = inputShape[0];
    unsigned int channelSize = inputShape[dataLayout.GetChannelsIndex()];

    std::vector<unsigned int> theBlockShape = m_Param.m_BlockShape;

    unsigned int overallSize = inBatchSize * inputShape[dataLayout.GetHeightIndex()]
                               * inputShape[dataLayout.GetWidthIndex()];

    std::vector<std::pair<unsigned int, unsigned int>> crops = m_Param.m_Crops;

    std::pair<unsigned int, unsigned int> yCrops = crops[0];
    std::pair<unsigned int, unsigned int> xCrops = crops[1];

    unsigned int inputHeight = inputShape[dataLayout.GetHeightIndex()];
    unsigned int outputHeight;

    unsigned int yCropsTotal = yCrops.first + yCrops.second;

    BOOST_ASSERT_MSG(yCropsTotal <= inputHeight,
                     "BatchToSpaceLayer: Overall height crop should be less than or equal to the input height.");

    unsigned int croppedHeight = inputHeight - yCropsTotal;

    if (theBlockShape.at(0) > 0)
    {
        outputHeight = theBlockShape.at(0) * croppedHeight;
    }
    else
    {
        outputHeight = croppedHeight;
    }

    unsigned int outputWidth;
    unsigned int inputWidth = inputShape[dataLayout.GetWidthIndex()];

    unsigned int xCropsTotal = xCrops.first + xCrops.second;

    BOOST_ASSERT_MSG(xCropsTotal <= inputWidth,
                     "BatchToSpaceLayer: Overall width crop should be less than or equal to the input width.");
    unsigned int croppedWidth = inputWidth - xCropsTotal;

    if (theBlockShape.at(1) > 0)
    {
        outputWidth = theBlockShape.at(1) * croppedWidth;
    }
    else
    {
        outputWidth = croppedWidth;
    }

    unsigned int outputBatchSize = overallSize / (outputHeight * outputWidth);

    if (dataLayout == DataLayout::NHWC)
    {
        return std::vector<TensorShape>({ TensorShape({ outputBatchSize, outputHeight, outputWidth, channelSize }) });
    }
    else
    {
        return std::vector<TensorShape>({ TensorShape({ outputBatchSize, channelSize, outputHeight, outputWidth }) });
    }
}
} // namespace armnn
