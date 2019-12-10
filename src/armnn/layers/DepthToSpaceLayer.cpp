//
// Copyright © 2019 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "DepthToSpaceLayer.hpp"
#include "LayerCloneBase.hpp"

#include <armnn/TypesUtils.hpp>

#include <armnnUtils/DataLayoutIndexed.hpp>

#include <backendsCommon/WorkloadData.hpp>
#include <backendsCommon/WorkloadFactory.hpp>

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

    return factory.CreateDepthToSpace(descriptor, PrepInfoAndDesc(descriptor));
}

DepthToSpaceLayer* DepthToSpaceLayer::Clone(Graph& graph) const
{
    return CloneBase<DepthToSpaceLayer>(graph, m_Param, GetName());
}

std::vector<TensorShape> DepthToSpaceLayer::InferOutputShapes(const std::vector<TensorShape>& inputShapes) const
{
    BOOST_ASSERT(inputShapes.size() == 1);

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

    std::vector<TensorShape> inferredShapes = InferOutputShapes({
        GetInputSlot(0).GetConnection()->GetTensorInfo().GetShape() });

    BOOST_ASSERT(inferredShapes.size() == 1);

    ConditionalThrowIfNotEqual<LayerValidationException>(
        "DepthToSpaceLayer: TensorShape set on OutputSlot[0] does not match the inferred shape.",
        GetOutputSlot(0).GetTensorInfo().GetShape(),
        inferredShapes[0]);
}

void DepthToSpaceLayer::Accept(ILayerVisitor& visitor) const
{
    visitor.VisitDepthToSpaceLayer(this, GetParameters(), GetName());
}

} // namespace armnn
