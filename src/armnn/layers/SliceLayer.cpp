//
// Copyright © 2019 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "SliceLayer.hpp"

#include "LayerCloneBase.hpp"

#include <armnn/TypesUtils.hpp>

#include <backendsCommon/WorkloadData.hpp>
#include <backendsCommon/WorkloadFactory.hpp>

#include <boost/numeric/conversion/cast.hpp>

namespace armnn
{

SliceLayer::SliceLayer(const SliceDescriptor& param, const char* name)
    : LayerWithParameters(1, 1, LayerType::Slice, param, name)
{
}

std::unique_ptr<IWorkload> SliceLayer::CreateWorkload(const IWorkloadFactory& factory) const
{
    SliceQueueDescriptor descriptor;
    return factory.CreateSlice(descriptor, PrepInfoAndDesc(descriptor));
}

SliceLayer* SliceLayer::Clone(Graph& graph) const
{
    return CloneBase<SliceLayer>(graph, m_Param, GetName());
}

void SliceLayer::ValidateTensorShapesFromInputs(ShapeInferenceMethod shapeInferenceMethod)
{
    IgnoreUnused(shapeInferenceMethod);

    VerifyLayerConnections(1, CHECK_LOCATION());

    auto inferredShapes = InferOutputShapes({ GetInputSlot(0).GetConnection()->GetTensorInfo().GetShape() });

    ARMNN_ASSERT(inferredShapes.size() == 1);

    ConditionalThrowIfNotEqual<LayerValidationException>(
            "SliceLayer: TensorShape set on OutputSlot[0] does not match the inferred shape.",
            GetOutputSlot(0).GetTensorInfo().GetShape(),
            inferredShapes[0]);
}

std::vector<TensorShape> SliceLayer::InferOutputShapes(const std::vector<TensorShape>& inputShapes) const
{
    IgnoreUnused(inputShapes);
    ARMNN_ASSERT(inputShapes.size() == 1);

    TensorShape outputShape(boost::numeric_cast<unsigned int>(m_Param.m_Size.size()), m_Param.m_Size.data());

    return std::vector<TensorShape>({ outputShape });
}

void SliceLayer::Accept(ILayerVisitor& visitor) const
{
    visitor.VisitSliceLayer(this, GetParameters(), GetName());
}

} // namespace armnn
