//
// Copyright © 2021 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//
#include "CastLayer.hpp"

#include "LayerCloneBase.hpp"
#include <armnn/TypesUtils.hpp>

#include <armnn/backends/WorkloadData.hpp>
#include <armnn/backends/WorkloadFactory.hpp>

namespace armnn
{

CastLayer::CastLayer(const char* name)
        : Layer(1, 1, LayerType::Cast, name)
{
}

std::unique_ptr<IWorkload> CastLayer::CreateWorkload(const IWorkloadFactory& factory) const
{
    CastQueueDescriptor descriptor;
    SetAdditionalInfo(descriptor);

    return factory.CreateWorkload(LayerType::Cast, descriptor, PrepInfoAndDesc(descriptor));
}

CastLayer* CastLayer::Clone(Graph& graph) const
{
    return CloneBase<CastLayer>(graph, GetName());
}

void CastLayer::ValidateTensorShapesFromInputs()
{
    VerifyLayerConnections(1, CHECK_LOCATION());

    const TensorShape& outputShape = GetOutputSlot(0).GetTensorInfo().GetShape();

    VerifyShapeInferenceType(outputShape, m_ShapeInferenceMethod);

    auto inferredShapes = InferOutputShapes({ GetInputSlot(0).GetConnection()->GetTensorInfo().GetShape() });

    ARMNN_ASSERT(inferredShapes.size() == 1);

    ValidateAndCopyShape(outputShape, inferredShapes[0], m_ShapeInferenceMethod, "CastLayer");
}

} // namespace armnn
