//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "SpaceToDepthLayer.hpp"
#include "LayerCloneBase.hpp"

#include <armnn/TypesUtils.hpp>
#include <armnn/utility/IgnoreUnused.hpp>
#include <armnnUtils/DataLayoutIndexed.hpp>

#include <backendsCommon/WorkloadData.hpp>
#include <backendsCommon/WorkloadFactory.hpp>

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

    return factory.CreateSpaceToDepth(descriptor, PrepInfoAndDesc(descriptor));
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

void SpaceToDepthLayer::ValidateTensorShapesFromInputs(ShapeInferenceMethod shapeInferenceMethod)
{
    IgnoreUnused(shapeInferenceMethod);

    VerifyLayerConnections(1, CHECK_LOCATION());

    std::vector<TensorShape> inferredShapes = InferOutputShapes({
        GetInputSlot(0).GetConnection()->GetTensorInfo().GetShape() });

    ARMNN_ASSERT(inferredShapes.size() == 1);

    ConditionalThrowIfNotEqual<LayerValidationException>(
        "SpaceToDepthLayer: TensorShape set on OutputSlot[0] does not match the inferred shape.",
        GetOutputSlot(0).GetTensorInfo().GetShape(),
        inferredShapes[0]);
}

void SpaceToDepthLayer::Accept(ILayerVisitor& visitor) const
{
    visitor.VisitSpaceToDepthLayer(this, GetParameters(), GetName());
}

} // namespace armnn
