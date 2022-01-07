//
// Copyright © 2017 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "QuantizeLayer.hpp"

#include "LayerCloneBase.hpp"

#include <armnn/ILayerVisitor.hpp>

namespace armnn
{

QuantizeLayer::QuantizeLayer(const char* name)
: Layer(1, 1, LayerType::Quantize, name)
{}

std::unique_ptr<IWorkload> QuantizeLayer::CreateWorkload(const IWorkloadFactory& factory) const
{
    QuantizeQueueDescriptor descriptor;
    SetAdditionalInfo(descriptor);

    WorkloadInfo info = PrepInfoAndDesc(descriptor);

    return factory.CreateWorkload(LayerType::Quantize, descriptor, info);
}

Layer* QuantizeLayer::Clone(Graph& graph) const
{
    QuantizeLayer* clone = CloneBase<QuantizeLayer>(graph, GetName());
    return clone;
}

void QuantizeLayer::ValidateTensorShapesFromInputs()
{
    VerifyLayerConnections(1, CHECK_LOCATION());

    const TensorShape& outputShape = GetOutputSlot(0).GetTensorInfo().GetShape();

    VerifyShapeInferenceType(outputShape, m_ShapeInferenceMethod);

    auto inferredShapes = InferOutputShapes({ GetInputSlot(0).GetConnection()->GetTensorInfo().GetShape() });

    ValidateAndCopyShape(outputShape, inferredShapes[0], m_ShapeInferenceMethod, "QuantizeLayer");
}

ARMNN_NO_DEPRECATE_WARN_BEGIN
void QuantizeLayer::Accept(ILayerVisitor& visitor) const
{
    visitor.VisitQuantizeLayer(this, GetName());
}
ARMNN_NO_DEPRECATE_WARN_END

} //namespace armnn