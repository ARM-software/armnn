//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//
#include "FakeQuantizationLayer.hpp"

#include "LayerCloneBase.hpp"

#include <armnn/TypesUtils.hpp>
#include <backendsCommon/WorkloadData.hpp>
#include <backendsCommon/WorkloadFactory.hpp>

namespace armnn
{

FakeQuantizationLayer::FakeQuantizationLayer(const FakeQuantizationDescriptor& param, const char* name)
: LayerWithParameters(1, 1, LayerType::FakeQuantization, param, name)
{
}

std::unique_ptr<IWorkload> FakeQuantizationLayer::CreateWorkload(const Graph& graph,
                                                                const IWorkloadFactory& factory) const
{
    FakeQuantizationQueueDescriptor descriptor;
    return factory.CreateFakeQuantization(descriptor, PrepInfoAndDesc(descriptor, graph) );
}

FakeQuantizationLayer* FakeQuantizationLayer::Clone(Graph& graph) const
{
    return CloneBase<FakeQuantizationLayer>(graph, m_Param, GetName());
}

void FakeQuantizationLayer::ValidateTensorShapesFromInputs()
{
    VerifyLayerConnections(1, CHECK_LOCATION());

    auto inferredShapes = InferOutputShapes({ GetInputSlot(0).GetConnection()->GetTensorInfo().GetShape() });

    BOOST_ASSERT(inferredShapes.size() == 1);

    ConditionalThrowIfNotEqual<LayerValidationException>(
        "FakeQuantizationLayer: TensorShape set on OutputSlot[0] does not match the inferred shape.",
        GetOutputSlot(0).GetTensorInfo().GetShape(),
        inferredShapes[0]);
}

void FakeQuantizationLayer::Accept(ILayerVisitor& visitor) const
{
    throw armnn::Exception("FakeQuantizationLayer should not appear in an input graph");
}

} // namespace armnn
