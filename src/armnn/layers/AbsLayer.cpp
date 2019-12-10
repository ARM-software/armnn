//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "AbsLayer.hpp"

#include "LayerCloneBase.hpp"

#include <armnn/TypesUtils.hpp>
#include <backendsCommon/WorkloadData.hpp>
#include <backendsCommon/WorkloadFactory.hpp>

namespace armnn
{

AbsLayer::AbsLayer(const char* name)
    : Layer(1, 1, LayerType::Abs, name)
{
}

std::unique_ptr<IWorkload> AbsLayer::CreateWorkload(const IWorkloadFactory& factory) const
{
    AbsQueueDescriptor descriptor;
    return factory.CreateAbs(descriptor, PrepInfoAndDesc(descriptor));
}

AbsLayer* AbsLayer::Clone(Graph& graph) const
{
    return CloneBase<AbsLayer>(graph, GetName());
}

void AbsLayer::ValidateTensorShapesFromInputs()
{
    VerifyLayerConnections(1, CHECK_LOCATION());

    auto inferredShapes = InferOutputShapes({ GetInputSlot(0).GetConnection()->GetTensorInfo().GetShape() });

    BOOST_ASSERT(inferredShapes.size() == 1);

    ConditionalThrowIfNotEqual<LayerValidationException>(
            "AbsLayer: TensorShape set on OutputSlot[0] does not match the inferred shape.",
            GetOutputSlot(0).GetTensorInfo().GetShape(),
            inferredShapes[0]);
}

void AbsLayer::Accept(ILayerVisitor& visitor) const
{
    visitor.VisitAbsLayer(this, GetName());
}

} // namespace armnn