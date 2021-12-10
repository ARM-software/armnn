//
// Copyright © 2017 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//
#include "MemImportLayer.hpp"

#include "LayerCloneBase.hpp"

#include <armnn/TypesUtils.hpp>
#include <armnn/backends/WorkloadData.hpp>
#include <armnn/backends/WorkloadFactory.hpp>
#include <backendsCommon/MemImportWorkload.hpp>

namespace armnn
{

MemImportLayer::MemImportLayer(const char* name)
    : Layer(1, 1, LayerType::MemImport, name)
{
}

MemImportLayer* MemImportLayer::Clone(Graph& graph) const
{
    return CloneBase<MemImportLayer>(graph, GetName());
}

std::unique_ptr<IWorkload> MemImportLayer::CreateWorkload(const IWorkloadFactory& factory) const
{
    IgnoreUnused(factory);
    MemImportQueueDescriptor descriptor;
    SetAdditionalInfo(descriptor);

    //This is different from other workloads. Does not get created by the workload factory.
    return std::make_unique<ImportMemGenericWorkload>(descriptor, PrepInfoAndDesc(descriptor));
}

void MemImportLayer::ValidateTensorShapesFromInputs()
{
    VerifyLayerConnections(1, CHECK_LOCATION());

    const TensorShape& outputShape = GetOutputSlot(0).GetTensorInfo().GetShape();

    VerifyShapeInferenceType(outputShape, m_ShapeInferenceMethod);

    auto inferredShapes = InferOutputShapes({ GetInputSlot(0).GetConnection()->GetTensorInfo().GetShape() });

    ARMNN_ASSERT(inferredShapes.size() == 1);

    ValidateAndCopyShape(outputShape, inferredShapes[0], m_ShapeInferenceMethod, "MemImportLayer");
}

ARMNN_NO_DEPRECATE_WARN_BEGIN
void MemImportLayer::Accept(ILayerVisitor& visitor) const
{
    IgnoreUnused(visitor);
    throw armnn::Exception("MemImportLayer should not appear in an input graph");
}
ARMNN_NO_DEPRECATE_WARN_END

void MemImportLayer::ExecuteStrategy(IStrategy& strategy) const
{
    IgnoreUnused(strategy);
    throw armnn::Exception("FakeQuantizationLayer should not appear in an input graph");
}

} // namespace armnn
