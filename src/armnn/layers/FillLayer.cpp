//
// Copyright © 2020 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//
#include "FillLayer.hpp"

#include "LayerCloneBase.hpp"

#include <armnn/TypesUtils.hpp>
#include <armnn/backends/WorkloadData.hpp>
#include <armnn/backends/WorkloadFactory.hpp>

namespace armnn
{

FillLayer::FillLayer(const FillDescriptor& param, const char* name)
    : LayerWithParameters(1, 1, LayerType::Fill, param, name)
{
}

std::unique_ptr<IWorkload> FillLayer::CreateWorkload(const IWorkloadFactory& factory) const
{
    FillQueueDescriptor descriptor;
    SetAdditionalInfo(descriptor);

    return factory.CreateWorkload(LayerType::Fill, descriptor, PrepInfoAndDesc(descriptor) );
}

FillLayer* FillLayer::Clone(Graph& graph) const
{
    return CloneBase<FillLayer>(graph, m_Param, GetName());
}

void FillLayer::ValidateTensorShapesFromInputs()
{
    VerifyLayerConnections(1, CHECK_LOCATION());

    const TensorShape& outputShape = GetOutputSlot(0).GetTensorInfo().GetShape();

    VerifyShapeInferenceType(outputShape, m_ShapeInferenceMethod);

    auto inferredShapes = InferOutputShapes({ GetInputSlot(0).GetConnection()->GetTensorInfo().GetShape() });

    ARMNN_ASSERT(inferredShapes.size() == 1);

    // Cannot validate the output shape from the input shape. but we can validate that the correct dims have been
    // inferred
    ConditionalThrowIfNotEqual<LayerValidationException>(
        "FillLayer: TensorShape set on OutputSlot[0] does not match the inferred shape.",
        GetOutputSlot(0).GetTensorInfo().GetNumDimensions(),
        inferredShapes[0][0]);
}

ARMNN_NO_DEPRECATE_WARN_BEGIN
void FillLayer::Accept(ILayerVisitor& visitor) const
{
    visitor.VisitFillLayer(this, GetParameters(), GetName());
}
ARMNN_NO_DEPRECATE_WARN_END

} // namespace armnn
