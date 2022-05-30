//
// Copyright © 2017 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "ResizeLayer.hpp"
#include "LayerCloneBase.hpp"

#include <armnn/TypesUtils.hpp>

#include <armnnUtils/DataLayoutIndexed.hpp>

#include <armnn/backends/WorkloadData.hpp>
#include <armnn/backends/WorkloadFactory.hpp>

using namespace armnnUtils;

namespace armnn
{

ResizeLayer::ResizeLayer(const ResizeDescriptor& param, const char* name)
    : LayerWithParameters(1, 1, LayerType::Resize, param, name)
{
}

std::unique_ptr<IWorkload> ResizeLayer::CreateWorkload(const IWorkloadFactory& factory) const
{
    ResizeQueueDescriptor descriptor;
    SetAdditionalInfo(descriptor);

    return factory.CreateWorkload(LayerType::Resize, descriptor, PrepInfoAndDesc(descriptor));
}

ResizeLayer* ResizeLayer::Clone(Graph& graph) const
{
    return CloneBase<ResizeLayer>(graph, m_Param, GetName());
}

std::vector<TensorShape> ResizeLayer::InferOutputShapes(const std::vector<TensorShape>& inputShapes) const
{
    ARMNN_ASSERT(inputShapes.size() == 1);

    const TensorShape& inputShape = inputShapes[0];
    const DataLayoutIndexed dimensionIndices = m_Param.m_DataLayout;

    unsigned int outWidth = m_Param.m_TargetWidth;
    unsigned int outHeight = m_Param.m_TargetHeight;
    unsigned int outChannels = inputShape[dimensionIndices.GetChannelsIndex()];
    unsigned int outBatch = inputShape[0];

    TensorShape tensorShape = m_Param.m_DataLayout == armnn::DataLayout::NHWC ?
        TensorShape( { outBatch, outHeight, outWidth, outChannels } ) :
        TensorShape( { outBatch, outChannels, outHeight, outWidth });

    if (m_Param.m_HalfPixelCenters && m_Param.m_AlignCorners)
    {
        throw LayerValidationException("ResizeLayer: AlignCorners cannot be true when HalfPixelCenters is true");
    }

    return std::vector<TensorShape>({ tensorShape });
}

void ResizeLayer::ValidateTensorShapesFromInputs()
{
    VerifyLayerConnections(1, CHECK_LOCATION());

    const TensorShape& outputShape = GetOutputSlot(0).GetTensorInfo().GetShape();

    VerifyShapeInferenceType(outputShape, m_ShapeInferenceMethod);

    auto inferredShapes = InferOutputShapes({ GetInputSlot(0).GetConnection()->GetTensorInfo().GetShape() });

    ARMNN_ASSERT(inferredShapes.size() == 1);

    ValidateAndCopyShape(outputShape, inferredShapes[0], m_ShapeInferenceMethod, "ResizeLayer");
}

void ResizeLayer::ExecuteStrategy(IStrategy& strategy) const
{
    strategy.ExecuteStrategy(this, GetParameters(), {}, GetName());
}

} // namespace armnn
