//
// Copyright © 2017 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "BatchToSpaceNdLayer.hpp"
#include "LayerCloneBase.hpp"
#include "LayerWithParameters.hpp"
#include "BatchToSpaceNdLayer.hpp"

#include <armnn/TypesUtils.hpp>

#include <armnnUtils/DataLayoutIndexed.hpp>

#include <armnn/backends/TensorHandle.hpp>
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

    auto inferredShapes = InferOutputShapes({GetInputSlot(0).GetConnection()->GetTensorInfo().GetShape()});

    ARMNN_ASSERT(inferredShapes.size() == 1);

    ValidateAndCopyShape(outputShape, inferredShapes[0], m_ShapeInferenceMethod, "BatchToSpaceNdLayer");
}

std::vector<TensorShape> BatchToSpaceNdLayer::InferOutputShapes(const std::vector<TensorShape>& inputShapes) const
{
    ARMNN_ASSERT(inputShapes.size() == 1);

    const TensorShape& inputShape = inputShapes[0];
    TensorShape outputShape(inputShape);

    unsigned int accumulatedBlockShape = std::accumulate(m_Param.m_BlockShape.begin(),
                                                         m_Param.m_BlockShape.end(),
                                                         1U,
                                                         std::multiplies<>());

    ARMNN_ASSERT(inputShape[0] % accumulatedBlockShape == 0);

    outputShape[0] = inputShape[0] / accumulatedBlockShape;

    DataLayoutIndexed dimensionIndices = m_Param.m_DataLayout;
    unsigned int heightIndex = dimensionIndices.GetHeightIndex();
    unsigned int widthIndex = dimensionIndices.GetWidthIndex();

    unsigned int heightCrop = m_Param.m_Crops[0].first + m_Param.m_Crops[0].second;
    unsigned int widthCrop = m_Param.m_Crops[1].first + m_Param.m_Crops[1].second;

    unsigned int outputHeight = inputShape[heightIndex] * m_Param.m_BlockShape[0];
    unsigned int outputWidth = inputShape[widthIndex] * m_Param.m_BlockShape[1];

    ARMNN_ASSERT_MSG(heightCrop <= outputHeight,
        "BatchToSpaceLayer: Overall height crop should be less than or equal to the uncropped output height.");

    ARMNN_ASSERT_MSG(widthCrop <= outputWidth,
        "BatchToSpaceLayer: Overall width crop should be less than or equal to the uncropped output width.");

    outputShape[heightIndex] = outputHeight - heightCrop;
    outputShape[widthIndex] = outputWidth - widthCrop;

    return std::vector<TensorShape>({ outputShape });
}

ARMNN_NO_DEPRECATE_WARN_BEGIN
void BatchToSpaceNdLayer::Accept(ILayerVisitor& visitor) const
{
    visitor.VisitBatchToSpaceNdLayer(this, GetParameters(), GetName());
}
ARMNN_NO_DEPRECATE_WARN_END

} // namespace armnn
