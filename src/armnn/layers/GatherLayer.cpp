//
// Copyright © 2017 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "GatherLayer.hpp"
#include "LayerCloneBase.hpp"

#include <armnn/TypesUtils.hpp>
#include <backendsCommon/WorkloadData.hpp>
#include <backendsCommon/WorkloadFactory.hpp>

namespace armnn
{

GatherLayer::GatherLayer(const GatherDescriptor& param, const char* name)
    : LayerWithParameters(2, 1, LayerType::Gather, param, name)
{
}

std::unique_ptr<IWorkload> GatherLayer::CreateWorkload(const armnn::IWorkloadFactory& factory) const
{
    GatherQueueDescriptor descriptor;
    SetAdditionalInfo(descriptor);

    return factory.CreateGather(descriptor, PrepInfoAndDesc(descriptor));
}

GatherLayer* GatherLayer::Clone(Graph& graph) const
{
    return CloneBase<GatherLayer>(graph, m_Param, GetName());
}

void GatherLayer::ValidateTensorShapesFromInputs()
{
    VerifyLayerConnections(2, CHECK_LOCATION());

    const TensorShape& outputShape = GetOutputSlot(0).GetTensorInfo().GetShape();

    VerifyShapeInferenceType(outputShape, m_ShapeInferenceMethod);

    const TensorInfo& params = GetInputSlot(0).GetConnection()->GetTensorInfo();
    const TensorInfo& indices = GetInputSlot(1).GetConnection()->GetTensorInfo();

    const unsigned int paramsDim = params.GetNumDimensions();
    const unsigned int indicesDim = indices.GetNumDimensions();
    const unsigned int outputDim = paramsDim - 1 + indicesDim;

    std::vector<unsigned int> dimSizes;

    unsigned int axis = static_cast<unsigned int>(m_Param.m_Axis);
    if (m_Param.m_Axis < 0)
    {
        int32_t  axis_aux = static_cast<int32_t>(paramsDim) + m_Param.m_Axis;
        axis = static_cast<unsigned int> (axis_aux);
    }

    for (unsigned int i = 0; i < axis; ++i)
    {
        dimSizes.push_back(params.GetShape()[i]);
    }
    for (unsigned int i = axis; i < indicesDim + axis; ++i)
    {
        dimSizes.push_back(indices.GetShape()[i - axis]);
    }
    for (unsigned int i = 1 + axis; i < paramsDim; ++i)
    {
        dimSizes.push_back(params.GetShape()[i]);
    }

    const TensorShape& inferredShape = TensorShape(outputDim, dimSizes.data());

    ValidateAndCopyShape(outputShape, inferredShape, m_ShapeInferenceMethod, "GatherLayer");
}

void GatherLayer::Accept(ILayerVisitor& visitor) const
{
    visitor.VisitGatherLayer(this, GetParameters(), GetName());
}

} // namespace armnn
