//
// Copyright © 2021 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "ChannelShuffleLayer.hpp"

#include "LayerCloneBase.hpp"

#include  <armnn/TypesUtils.hpp>

#include <armnn/backends/Workload.hpp>
#include <armnn/backends/WorkloadFactory.hpp>

namespace armnn
{
ChannelShuffleLayer::ChannelShuffleLayer(const ChannelShuffleDescriptor& param, const char* name)
: LayerWithParameters(1, 1, LayerType::ChannelShuffle, param, name)
{
}

std::unique_ptr<IWorkload> ChannelShuffleLayer::CreateWorkload(const IWorkloadFactory &factory) const
{
    ChannelShuffleQueueDescriptor descriptor;
    SetAdditionalInfo(descriptor);

    return factory.CreateWorkload(LayerType::ChannelShuffle, descriptor, PrepInfoAndDesc(descriptor));
}

ChannelShuffleLayer* ChannelShuffleLayer::Clone(Graph& graph) const
{
    return CloneBase<ChannelShuffleLayer>(graph, m_Param, GetName());
}

void ChannelShuffleLayer::ValidateTensorShapesFromInputs()
{
    VerifyLayerConnections(1, CHECK_LOCATION());

    const TensorShape& outputShape = GetOutputSlot(0).GetTensorInfo().GetShape();

    VerifyShapeInferenceType(outputShape, m_ShapeInferenceMethod);

    auto inferredShapes = Layer::InferOutputShapes({GetInputSlot(0).GetConnection()->GetTensorInfo().GetShape() });

    ARMNN_ASSERT(inferredShapes.size() == 1);

    ValidateAndCopyShape(outputShape, inferredShapes[0], m_ShapeInferenceMethod, "ChannelShuffleLayer");
}

}