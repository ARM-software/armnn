//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//
#include "MemCopyLayer.hpp"

#include "LayerCloneBase.hpp"

#include <armnn/TypesUtils.hpp>
#include <backendsCommon/WorkloadData.hpp>
#include <backendsCommon/WorkloadFactory.hpp>
#include <backendsCommon/MemCopyWorkload.hpp>

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

std::unique_ptr<IWorkload> MemCopyLayer::CreateWorkload(const Graph& graph, const IWorkloadFactory& factory) const
{
    MemCopyQueueDescriptor descriptor;

    //This is different from other workloads. Does not get created by the workload factory.
    return std::make_unique<CopyMemGenericWorkload>(descriptor, PrepInfoAndDesc(descriptor, graph));
}

void MemCopyLayer::ValidateTensorShapesFromInputs()
{
    VerifyLayerConnections(1, CHECK_LOCATION());

    auto inferredShapes = InferOutputShapes({ GetInputSlot(0).GetConnection()->GetTensorInfo().GetShape() });

    BOOST_ASSERT(inferredShapes.size() == 1);

    ConditionalThrowIfNotEqual<LayerValidationException>(
        "MemCopyLayer: TensorShape set on OutputSlot[0] does not match the inferred shape.",
        GetOutputSlot(0).GetTensorInfo().GetShape(),
        inferredShapes[0]);
}

void MemCopyLayer::Accept(ILayerVisitor& visitor) const
{
    throw armnn::Exception("MemCopyLayer should not appear in an input graph");
}

} // namespace armnn
