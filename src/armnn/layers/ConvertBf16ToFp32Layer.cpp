//
// Copyright © 2020 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "ConvertBf16ToFp32Layer.hpp"
#include "LayerCloneBase.hpp"

#include <armnn/TypesUtils.hpp>

#include <backendsCommon/WorkloadData.hpp>
#include <backendsCommon/WorkloadFactory.hpp>

namespace armnn
{

ConvertBf16ToFp32Layer::ConvertBf16ToFp32Layer(const char* name)
    : Layer(1, 1, LayerType::ConvertBf16ToFp32, name)
{
}

std::unique_ptr<IWorkload> ConvertBf16ToFp32Layer::CreateWorkload(const IWorkloadFactory& factory) const
{
    ConvertBf16ToFp32QueueDescriptor descriptor;
    return factory.CreateConvertBf16ToFp32(descriptor, PrepInfoAndDesc(descriptor));
}

ConvertBf16ToFp32Layer* ConvertBf16ToFp32Layer::Clone(Graph& graph) const
{
    return CloneBase<ConvertBf16ToFp32Layer>(graph, GetName());
}

void ConvertBf16ToFp32Layer::ValidateTensorShapesFromInputs()
{
    VerifyLayerConnections(1, CHECK_LOCATION());

    auto inferredShapes = InferOutputShapes({ GetInputSlot(0).GetConnection()->GetTensorInfo().GetShape() });

    ARMNN_ASSERT(inferredShapes.size() == 1);

    ConditionalThrowIfNotEqual<LayerValidationException>(
        "ConvertBf16ToFp32Layer: TensorShape set on OutputSlot[0] does not match the inferred shape.",
        GetOutputSlot(0).GetTensorInfo().GetShape(),
        inferredShapes[0]);
}

void ConvertBf16ToFp32Layer::Accept(ILayerVisitor& visitor) const
{
    // these conversion layers are only inserted by the
    // optimizer and so will never be in an input graph.
    IgnoreUnused(visitor);
    throw armnn::Exception("ConvertBf16ToFp32Layer should never appear in an input graph");
}

} // namespace armnn
