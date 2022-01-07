//
// Copyright © 2020 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "RankLayer.hpp"

#include "LayerCloneBase.hpp"

#include <armnn/backends/WorkloadData.hpp>
#include <armnn/backends/WorkloadFactory.hpp>

namespace armnn
{

RankLayer::RankLayer(const char* name)
        : Layer(1, 1, LayerType::Rank, name)
{}

std::unique_ptr<IWorkload> RankLayer::CreateWorkload(const IWorkloadFactory& factory) const
{
    RankQueueDescriptor descriptor;
    SetAdditionalInfo(descriptor);

    return factory.CreateWorkload(LayerType::Rank, descriptor, PrepInfoAndDesc(descriptor));
}

Layer* RankLayer::Clone(Graph& graph) const
{
    RankLayer* clone = CloneBase<RankLayer>(graph, GetName());
    return clone;
}

void RankLayer::ValidateTensorShapesFromInputs()
{
    VerifyLayerConnections(1, CHECK_LOCATION());

    const TensorShape& outputShape = GetOutputSlot(0).GetTensorInfo().GetShape();
    const TensorShape inferredShape = TensorShape(Dimensionality::Scalar);

    VerifyShapeInferenceType(outputShape, m_ShapeInferenceMethod);
    ValidateAndCopyShape(outputShape, inferredShape, m_ShapeInferenceMethod, "RankLayer");
}

ARMNN_NO_DEPRECATE_WARN_BEGIN
void RankLayer::Accept(ILayerVisitor& visitor) const
{
    visitor.VisitRankLayer(this, GetName());
}
ARMNN_NO_DEPRECATE_WARN_END

void RankLayer::ExecuteStrategy(IStrategy& strategy) const
{
    strategy.ExecuteStrategy(this, BaseDescriptor(), {}, GetName());
}

} //namespace armnn