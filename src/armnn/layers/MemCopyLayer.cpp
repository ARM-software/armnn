//
// Copyright © 2017 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//
#include "MemCopyLayer.hpp"

#include "LayerCloneBase.hpp"

#include <armnn/TypesUtils.hpp>
#include <armnn/backends/WorkloadData.hpp>
#include <armnn/backends/WorkloadFactory.hpp>
#include <armnn/backends/MemCopyWorkload.hpp>

namespace armnn
{

MemCopyLayer::MemCopyLayer(const char* name)
    : Layer(1, 1, LayerType::MemCopy, name)
{
}

MemCopyLayer* MemCopyLayer::Clone(Graph& graph) const
{
    return CloneBase<MemCopyLayer>(graph, GetName());
}

std::unique_ptr<IWorkload> MemCopyLayer::CreateWorkload(const IWorkloadFactory& factory) const
{
    IgnoreUnused(factory);
    MemCopyQueueDescriptor descriptor;
    SetAdditionalInfo(descriptor);

    //This is different from other workloads. Does not get created by the workload factory.
    return std::make_unique<CopyMemGenericWorkload>(descriptor, PrepInfoAndDesc(descriptor));
}

void MemCopyLayer::ValidateTensorShapesFromInputs()
{
    VerifyLayerConnections(1, CHECK_LOCATION());

    const TensorShape& outputShape = GetOutputSlot(0).GetTensorInfo().GetShape();

    VerifyShapeInferenceType(outputShape, m_ShapeInferenceMethod);

    auto inferredShapes = InferOutputShapes({ GetInputSlot(0).GetConnection()->GetTensorInfo().GetShape() });

    ARMNN_ASSERT(inferredShapes.size() == 1);

    ValidateAndCopyShape(outputShape, inferredShapes[0], m_ShapeInferenceMethod, "MemCopyLayer");
}

void MemCopyLayer::ExecuteStrategy(IStrategy& strategy) const
{
    strategy.ExecuteStrategy(this, GetParameters(), {}, GetName());
}

} // namespace armnn
