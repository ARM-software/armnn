//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//
#include "ResizeLayer.hpp"

#include "LayerCloneBase.hpp"

#include <armnn/TypesUtils.hpp>

#include <backendsCommon/WorkloadData.hpp>
#include <backendsCommon/WorkloadFactory.hpp>

#include <DataLayoutIndexed.hpp>

using namespace armnnUtils;

namespace armnn
{

ResizeLayer::ResizeLayer(const ResizeDescriptor& param, const char* name)
    : LayerWithParameters(1, 1, LayerType::Resize, param, name)
{
}

std::unique_ptr<IWorkload> ResizeLayer::CreateWorkload(const Graph& graph,
                                                       const IWorkloadFactory& factory) const
{
    ResizeQueueDescriptor descriptor;
    return factory.CreateResize(descriptor, PrepInfoAndDesc(descriptor, graph));
}

ResizeLayer* ResizeLayer::Clone(Graph& graph) const
{
    return CloneBase<ResizeLayer>(graph, m_Param, GetName());
}

std::vector<TensorShape> ResizeLayer::InferOutputShapes(const std::vector<TensorShape>& inputShapes) const
{
    BOOST_ASSERT(inputShapes.size() == 1);

    const TensorShape& inputShape = inputShapes[0];
    const DataLayoutIndexed dimensionIndices = m_Param.m_DataLayout;

    unsigned int outWidth = m_Param.m_TargetWidth;
    unsigned int outHeight = m_Param.m_TargetHeight;
    unsigned int outChannels = inputShape[dimensionIndices.GetChannelsIndex()];
    unsigned int outBatch = inputShape[0];

    TensorShape tensorShape = m_Param.m_DataLayout == armnn::DataLayout::NHWC ?
        TensorShape( { outBatch, outHeight, outWidth, outChannels } ) :
        TensorShape( { outBatch, outChannels, outHeight, outWidth });

    return std::vector<TensorShape>({ tensorShape });
}

void ResizeLayer::ValidateTensorShapesFromInputs()
{
    VerifyLayerConnections(1, CHECK_LOCATION());

    auto inferredShapes = InferOutputShapes({ GetInputSlot(0).GetConnection()->GetTensorInfo().GetShape() });

    BOOST_ASSERT(inferredShapes.size() == 1);

    ConditionalThrowIfNotEqual<LayerValidationException>(
        "ResizeLayer: TensorShape set on OutputSlot[0] does not match the inferred shape.",
        GetOutputSlot(0).GetTensorInfo().GetShape(),
        inferredShapes[0]);
}

void ResizeLayer::Accept(ILayerVisitor& visitor) const
{
    visitor.VisitResizeLayer(this, GetParameters(), GetName());
}

} // namespace armnn
