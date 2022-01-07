//
// Copyright © 2017 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "SpaceToDepthLayer.hpp"
#include "LayerCloneBase.hpp"

#include <armnn/TypesUtils.hpp>
#include <armnn/utility/IgnoreUnused.hpp>
#include <armnnUtils/DataLayoutIndexed.hpp>

#include <armnn/backends/WorkloadData.hpp>
#include <armnn/backends/WorkloadFactory.hpp>

#include <numeric>

using namespace armnnUtils;

namespace armnn
{

SpaceToDepthLayer::SpaceToDepthLayer(const SpaceToDepthDescriptor param, const char* name)
    : LayerWithParameters(1, 1, LayerType::SpaceToDepth, param, name)
{}

std::unique_ptr<IWorkload> SpaceToDepthLayer::CreateWorkload(const IWorkloadFactory& factory) const
{
    SpaceToDepthQueueDescriptor descriptor;
    descriptor.m_Parameters.m_BlockSize  = m_Param.m_BlockSize;
    descriptor.m_Parameters.m_DataLayout = m_Param.m_DataLayout;

    SetAdditionalInfo(descriptor);

    return factory.CreateWorkload(LayerType::SpaceToDepth, descriptor, PrepInfoAndDesc(descriptor));
}

SpaceToDepthLayer* SpaceToDepthLayer::Clone(Graph& graph) const
{
    IgnoreUnused(graph);
    return CloneBase<SpaceToDepthLayer>(graph, m_Param, GetName());
}

std::vector<TensorShape> SpaceToDepthLayer::InferOutputShapes(const std::vector<TensorShape>& inputShapes) const
{
    ARMNN_ASSERT(inputShapes.size() == 1);

    TensorShape inputShape = inputShapes[0];
    TensorShape outputShape(inputShape);

    DataLayoutIndexed dimensionIndices{m_Param.m_DataLayout};
    unsigned int hIndex = dimensionIndices.GetHeightIndex();
    unsigned int wIndex = dimensionIndices.GetWidthIndex();
    unsigned int cIndex = dimensionIndices.GetChannelsIndex();

    outputShape[hIndex] = inputShape[hIndex] / m_Param.m_BlockSize;
    outputShape[wIndex] = inputShape[wIndex] / m_Param.m_BlockSize;

    outputShape[cIndex] = inputShape[cIndex] * m_Param.m_BlockSize * m_Param.m_BlockSize;

    return std::vector<TensorShape>({ outputShape });
}

void SpaceToDepthLayer::ValidateTensorShapesFromInputs()
{
    VerifyLayerConnections(1, CHECK_LOCATION());

    const TensorShape& outputShape = GetOutputSlot(0).GetTensorInfo().GetShape();

    VerifyShapeInferenceType(outputShape, m_ShapeInferenceMethod);

    std::vector<TensorShape> inferredShapes = InferOutputShapes({
        GetInputSlot(0).GetConnection()->GetTensorInfo().GetShape() });

    ARMNN_ASSERT(inferredShapes.size() == 1);

    ValidateAndCopyShape(outputShape, inferredShapes[0], m_ShapeInferenceMethod, "SpaceToDepthLayer");
}

ARMNN_NO_DEPRECATE_WARN_BEGIN
void SpaceToDepthLayer::Accept(ILayerVisitor& visitor) const
{
    visitor.VisitSpaceToDepthLayer(this, GetParameters(), GetName());
}
ARMNN_NO_DEPRECATE_WARN_END

} // namespace armnn
