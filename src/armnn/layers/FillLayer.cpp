//
// Copyright © 2020 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//
#include "FillLayer.hpp"

#include "LayerCloneBase.hpp"

#include <armnn/TypesUtils.hpp>
#include <backendsCommon/WorkloadData.hpp>
#include <backendsCommon/WorkloadFactory.hpp>

namespace armnn
{

FillLayer::FillLayer(const FillDescriptor& param, const char* name)
    : LayerWithParameters(1, 1, LayerType::Fill, param, name)
{
}

std::unique_ptr<IWorkload> FillLayer::CreateWorkload(const IWorkloadFactory& factory) const
{
    FillQueueDescriptor descriptor;
    return factory.CreateFill(descriptor, PrepInfoAndDesc(descriptor) );
}

FillLayer* FillLayer::Clone(Graph& graph) const
{
    return CloneBase<FillLayer>(graph, m_Param, GetName());
}

void FillLayer::ValidateTensorShapesFromInputs(ShapeInferenceMethod shapeInferenceMethod)
{
    IgnoreUnused(shapeInferenceMethod);

    VerifyLayerConnections(1, CHECK_LOCATION());

    auto inferredShapes = InferOutputShapes( { GetInputSlot(0).GetConnection()->GetTensorInfo().GetShape() });

    ARMNN_ASSERT(inferredShapes.size() == 1);

    // Cannot validate the output shape from the input shape. but we can validate that the correct dims have been
    // inferred
    ConditionalThrowIfNotEqual<LayerValidationException>(
        "FillLayer: TensorShape set on OutputSlot[0] does not match the inferred shape.",
        GetOutputSlot(0).GetTensorInfo().GetNumDimensions(),
        inferredShapes[0][0]);
}

void FillLayer::Accept(ILayerVisitor& visitor) const
{
    visitor.VisitFillLayer(this, GetParameters(), GetName());
}

} // namespace armnn
