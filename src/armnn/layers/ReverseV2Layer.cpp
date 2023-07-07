//
// Copyright © 2023 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "ReverseV2Layer.hpp"

#include <armnn/backends/WorkloadFactory.hpp>
#include "layers/LayerCloneBase.hpp"

namespace armnn
{
ReverseV2Layer::ReverseV2Layer(const armnn::ReverseV2Descriptor &param, const char *name)
    : LayerWithParameters(1, 1, LayerType::ReverseV2, param, name)
{}

std::unique_ptr<IWorkload> ReverseV2Layer::CreateWorkload(const armnn::IWorkloadFactory &factory) const
{
    ReverseV2QueueDescriptor descriptor;
    SetAdditionalInfo(descriptor);

    return factory.CreateWorkload(LayerType::ReverseV2, descriptor, PrepInfoAndDesc(descriptor));
}

ReverseV2Layer* ReverseV2Layer::Clone(armnn::Graph &graph) const
{
    auto layer = CloneBase<ReverseV2Layer>(graph, m_Param, GetName());

    return std::move(layer);
}

/// Use the default Layer::InferOutputShape method

void ReverseV2Layer::ValidateTensorShapesFromInputs()
{
    VerifyLayerConnections(1, CHECK_LOCATION());

    const TensorShape& outputShape = GetOutputSlot(0).GetTensorInfo().GetShape();

    VerifyShapeInferenceType(outputShape, m_ShapeInferenceMethod);

    auto inferredShapes = InferOutputShapes({
        GetInputSlot(0).GetTensorInfo().GetShape() });

    ARMNN_ASSERT(inferredShapes.size() == 1);

    ValidateAndCopyShape(outputShape, inferredShapes[0], m_ShapeInferenceMethod, "ReverseV2Layer");
}

}