//
// Copyright © 2018-2023 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "BatchToSpaceNdLayer.hpp"
#include "LayerCloneBase.hpp"

#include <armnn/backends/WorkloadData.hpp>
#include <armnn/backends/WorkloadFactory.hpp>

#include <numeric>

using namespace armnnUtils;

namespace armnn
{

BatchToSpaceNdLayer::BatchToSpaceNdLayer(const armnn::BatchToSpaceNdDescriptor& param, const char* name)
    : LayerWithParameters(1, 1, LayerType::BatchToSpaceNd, param, name)
{
}

std::unique_ptr<IWorkload> BatchToSpaceNdLayer::CreateWorkload(const IWorkloadFactory& factory) const
{
    BatchToSpaceNdQueueDescriptor descriptor;
    SetAdditionalInfo(descriptor);

    return factory.CreateWorkload(LayerType::BatchToSpaceNd, descriptor, PrepInfoAndDesc(descriptor));
}

BatchToSpaceNdLayer* BatchToSpaceNdLayer::Clone(Graph& graph) const
{
    auto layer = CloneBase<BatchToSpaceNdLayer>(graph, m_Param, GetName());
    return std::move(layer);
}

void BatchToSpaceNdLayer::ValidateTensorShapesFromInputs()
{
    VerifyLayerConnections(1, CHECK_LOCATION());

    const TensorShape &outputShape = GetOutputSlot(0).GetTensorInfo().GetShape();

    VerifyShapeInferenceType(outputShape, m_ShapeInferenceMethod);

    auto inferredShapes = InferOutputShapes({GetInputSlot(0).GetTensorInfo().GetShape()});

    ARMNN_ASSERT(inferredShapes.size() == 1);

    ValidateAndCopyShape(outputShape, inferredShapes[0], m_ShapeInferenceMethod, "BatchToSpaceNdLayer");
}

std::vector<TensorShape> BatchToSpaceNdLayer::InferOutputShapes(const std::vector<TensorShape>& inputShapes) const
{
    const TensorShape& inputShape = inputShapes[0];
    TensorShape outputShape(inputShape);

    unsigned int accumulatedBlockShape = std::accumulate(m_Param.m_BlockShape.begin(),
                                                         m_Param.m_BlockShape.end(),
                                                         1U,
                                                         std::multiplies<>());
    outputShape[0] = (inputShape[0] / accumulatedBlockShape) < 1 ? 1 : (inputShape[0] / accumulatedBlockShape) ;

    // In a 4D tensor, there will be 2 spatialDimensions (H and W), and the for loop will run twice.
    // In a 3D tensor, there will be 1 spatialDimensions, and the for loop will run once.
    unsigned int firstSpatialDimension = m_Param.m_DataLayout == DataLayout::NCHW ? 2 : 1;
    for (unsigned int i = 0; i < m_Param.m_BlockShape.size(); ++i)
    {
        unsigned int spatialDimension = firstSpatialDimension + i;
        unsigned int cropSize = m_Param.m_Crops[i].first + m_Param.m_Crops[i].second;
        unsigned int outputSize = inputShape[spatialDimension] * m_Param.m_BlockShape[i];
        outputShape[spatialDimension] = outputSize - cropSize;
    }

    return std::vector<TensorShape>({ outputShape });
}

void BatchToSpaceNdLayer::ExecuteStrategy(IStrategy& strategy) const
{
    strategy.ExecuteStrategy(this, GetParameters(), {}, GetName());
}

} // namespace armnn
