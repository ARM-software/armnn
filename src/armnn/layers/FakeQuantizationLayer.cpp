//
// Copyright © 2017 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//
#include "FakeQuantizationLayer.hpp"

#include "LayerCloneBase.hpp"

#include <armnn/TypesUtils.hpp>
#include <armnn/backends/WorkloadData.hpp>
#include <armnn/backends/WorkloadFactory.hpp>

namespace armnn
{

FakeQuantizationLayer::FakeQuantizationLayer(const FakeQuantizationDescriptor& param, const char* name)
: LayerWithParameters(1, 1, LayerType::FakeQuantization, param, name)
{
}

std::unique_ptr<IWorkload> FakeQuantizationLayer::CreateWorkload(const IWorkloadFactory& factory) const
{
    FakeQuantizationQueueDescriptor descriptor;
    SetAdditionalInfo(descriptor);

    return factory.CreateWorkload(LayerType::FakeQuantization, descriptor, PrepInfoAndDesc(descriptor) );
}

FakeQuantizationLayer* FakeQuantizationLayer::Clone(Graph& graph) const
{
    return CloneBase<FakeQuantizationLayer>(graph, m_Param, GetName());
}

void FakeQuantizationLayer::ValidateTensorShapesFromInputs()
{
    VerifyLayerConnections(1, CHECK_LOCATION());

    const TensorShape& outputShape = GetOutputSlot(0).GetTensorInfo().GetShape();

    VerifyShapeInferenceType(outputShape, m_ShapeInferenceMethod);

    auto inferredShapes = InferOutputShapes({ GetInputSlot(0).GetConnection()->GetTensorInfo().GetShape() });

    ARMNN_ASSERT(inferredShapes.size() == 1);

    ValidateAndCopyShape(outputShape, inferredShapes[0], m_ShapeInferenceMethod, "FakeQuantizationLayer");
}

void FakeQuantizationLayer::ExecuteStrategy(IStrategy& strategy) const
{
    IgnoreUnused(strategy);
    throw armnn::Exception("FakeQuantizationLayer should not appear in an input graph");
}

} // namespace armnn
