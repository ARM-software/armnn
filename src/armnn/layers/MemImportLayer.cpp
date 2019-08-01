//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//
#include "MemImportLayer.hpp"

#include "LayerCloneBase.hpp"

#include <armnn/TypesUtils.hpp>
#include <backendsCommon/WorkloadData.hpp>
#include <backendsCommon/WorkloadFactory.hpp>
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

std::unique_ptr<IWorkload> MemImportLayer::CreateWorkload(const Graph& graph, const IWorkloadFactory& factory) const
{
    MemImportQueueDescriptor descriptor;

    //This is different from other workloads. Does not get created by the workload factory.
    return std::make_unique<ImportMemGenericWorkload>(descriptor, PrepInfoAndDesc(descriptor, graph));
}

void MemImportLayer::ValidateTensorShapesFromInputs()
{
    VerifyLayerConnections(1, CHECK_LOCATION());

    auto inferredShapes = InferOutputShapes({ GetInputSlot(0).GetConnection()->GetTensorInfo().GetShape() });

    BOOST_ASSERT(inferredShapes.size() == 1);

    ConditionalThrowIfNotEqual<LayerValidationException>(
        "MemImportLayer: TensorShape set on OutputSlot[0] does not match the inferred shape.",
        GetOutputSlot(0).GetTensorInfo().GetShape(),
        inferredShapes[0]);
}

void MemImportLayer::Accept(ILayerVisitor& visitor) const
{
    throw armnn::Exception("MemImportLayer should not appear in an input graph");
}

} // namespace armnn
