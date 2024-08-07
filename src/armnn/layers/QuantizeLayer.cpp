//
// Copyright © 2019-2023 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "QuantizeLayer.hpp"

#include "LayerCloneBase.hpp"

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

    auto inferredShapes = InferOutputShapes({ GetInputSlot(0).GetTensorInfo().GetShape() });

    ValidateAndCopyShape(outputShape, inferredShapes[0], m_ShapeInferenceMethod, "QuantizeLayer");
}

void QuantizeLayer::ExecuteStrategy(IStrategy& strategy) const
{
    strategy.ExecuteStrategy(this, GetParameters(), {}, GetName());
}

} //namespace armnn