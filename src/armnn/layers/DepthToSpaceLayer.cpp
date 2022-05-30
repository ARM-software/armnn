//
// Copyright © 2019 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "DepthToSpaceLayer.hpp"
#include "LayerCloneBase.hpp"

#include <armnn/TypesUtils.hpp>

#include <armnnUtils/DataLayoutIndexed.hpp>

#include <armnn/backends/WorkloadData.hpp>
#include <armnn/backends/WorkloadFactory.hpp>

#include <numeric>

namespace armnn
{

DepthToSpaceLayer::DepthToSpaceLayer(const DepthToSpaceDescriptor& param, const char* name)
    : LayerWithParameters(1, 1, LayerType::DepthToSpace, param, name)
{}

std::unique_ptr<IWorkload> DepthToSpaceLayer::CreateWorkload(const IWorkloadFactory& factory) const
{
    DepthToSpaceQueueDescriptor descriptor;
    descriptor.m_Parameters.m_BlockSize  = m_Param.m_BlockSize;
    descriptor.m_Parameters.m_DataLayout = m_Param.m_DataLayout;

    SetAdditionalInfo(descriptor);

    return factory.CreateWorkload(LayerType::DepthToSpace, descriptor, PrepInfoAndDesc(descriptor));
}

DepthToSpaceLayer* DepthToSpaceLayer::Clone(Graph& graph) const
{
    return CloneBase<DepthToSpaceLayer>(graph, m_Param, GetName());
}

std::vector<TensorShape> DepthToSpaceLayer::InferOutputShapes(const std::vector<TensorShape>& inputShapes) const
{
    ARMNN_ASSERT(inputShapes.size() == 1);

    TensorShape inputShape = inputShapes[0];
    TensorShape outputShape(inputShape);

    armnnUtils::DataLayoutIndexed dimensionIndices(m_Param.m_DataLayout);

    unsigned int hIndex = dimensionIndices.GetHeightIndex();
    unsigned int wIndex = dimensionIndices.GetWidthIndex();
    unsigned int cIndex = dimensionIndices.GetChannelsIndex();

    outputShape[hIndex] = inputShape[hIndex] * m_Param.m_BlockSize;
    outputShape[wIndex] = inputShape[wIndex] * m_Param.m_BlockSize;

    outputShape[cIndex] = inputShape[cIndex] / (m_Param.m_BlockSize * m_Param.m_BlockSize);

    return std::vector<TensorShape>({ outputShape });
}

void DepthToSpaceLayer::ValidateTensorShapesFromInputs()
{
    VerifyLayerConnections(1, CHECK_LOCATION());

    const TensorShape& outputShape = GetOutputSlot(0).GetTensorInfo().GetShape();

    VerifyShapeInferenceType(outputShape, m_ShapeInferenceMethod);

    std::vector<TensorShape> inferredShapes = InferOutputShapes({
        GetInputSlot(0).GetConnection()->GetTensorInfo().GetShape() });

    ARMNN_ASSERT(inferredShapes.size() == 1);

    ValidateAndCopyShape(outputShape, inferredShapes[0], m_ShapeInferenceMethod, "DepthToSpaceLayer");
}

void DepthToSpaceLayer::ExecuteStrategy(IStrategy& strategy) const
{
    strategy.ExecuteStrategy(this, GetParameters(), {}, GetName());
}

} // namespace armnn
