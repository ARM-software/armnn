//
// Copyright © 2017 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "AbsLayer.hpp"

#include "LayerCloneBase.hpp"

#include <armnn/TypesUtils.hpp>
#include <backendsCommon/WorkloadData.hpp>
#include <backendsCommon/WorkloadFactory.hpp>

namespace armnn
{

AbsLayer::AbsLayer(const char* name)
    : Layer(1, 1, LayerType::Abs, name)
{
}

std::unique_ptr<IWorkload> AbsLayer::CreateWorkload(const IWorkloadFactory& factory) const
{
    AbsQueueDescriptor descriptor;
    return factory.CreateAbs(descriptor, PrepInfoAndDesc(descriptor));
}

AbsLayer* AbsLayer::Clone(Graph& graph) const
{
    return CloneBase<AbsLayer>(graph, GetName());
}

void AbsLayer::ValidateTensorShapesFromInputs(ShapeInferenceMethod shapeInferenceMethod)
{
    VerifyLayerConnections(1, CHECK_LOCATION());
    const TensorShape& outputShape = GetOutputSlot(0).GetTensorInfo().GetShape();

    VerifyShapeInferenceType(outputShape, shapeInferenceMethod);

    auto inferredShapes = InferOutputShapes({ GetInputSlot(0).GetConnection()->GetTensorInfo().GetShape() });

    ARMNN_ASSERT(inferredShapes.size() == 1);

    ValidateAndCopyShape(outputShape, inferredShapes[0], shapeInferenceMethod, "AbsLayer");
}

void AbsLayer::Accept(ILayerVisitor& visitor) const
{
    visitor.VisitAbsLayer(this, GetName());
}

} // namespace armnn