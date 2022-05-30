//
// Copyright © 2017 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "GatherLayer.hpp"
#include "LayerCloneBase.hpp"

#include <armnn/TypesUtils.hpp>
#include <armnn/backends/WorkloadData.hpp>
#include <armnn/backends/WorkloadFactory.hpp>

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

    return factory.CreateWorkload(LayerType::Gather, descriptor, PrepInfoAndDesc(descriptor));
}

GatherLayer* GatherLayer::Clone(Graph& graph) const
{
    return CloneBase<GatherLayer>(graph, m_Param, GetName());
}

std::vector<TensorShape> GatherLayer::InferOutputShapes(const std::vector<TensorShape>& inputShapes) const
{
    ARMNN_ASSERT(inputShapes.size() == 2);
    const TensorShape& params = inputShapes[0];
    const TensorShape& indices = inputShapes[1];

    if (indices.GetDimensionality() == Dimensionality::Scalar && indices.GetNumDimensions() == 1)
    {
         return std::vector<TensorShape>({ TensorShape(Dimensionality::Scalar)});
    }

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
        dimSizes.push_back(params[i]);
    }
    for (unsigned int i = axis; i < indicesDim + axis; ++i)
    {
        dimSizes.push_back(indices[i - axis]);
    }
    for (unsigned int i = 1 + axis; i < paramsDim; ++i)
    {
        dimSizes.push_back(params[i]);
    }

    return std::vector<TensorShape>({ TensorShape({outputDim, dimSizes.data()})});
}

void GatherLayer::ValidateTensorShapesFromInputs()
{
    VerifyLayerConnections(2, CHECK_LOCATION());

    const TensorShape& outputShape = GetOutputSlot(0).GetTensorInfo().GetShape();

    VerifyShapeInferenceType(outputShape, m_ShapeInferenceMethod);

    std::vector<TensorShape> inferredShapes = InferOutputShapes(
            {GetInputSlot(0).GetConnection()->GetTensorInfo().GetShape(),
             GetInputSlot(1).GetConnection()->GetTensorInfo().GetShape()});
    ARMNN_ASSERT(inferredShapes.size() == 1);
    ARMNN_ASSERT(inferredShapes[0].GetDimensionality() == Dimensionality::Specified ||
                 inferredShapes[0].GetDimensionality() == Dimensionality::Scalar);

    ValidateAndCopyShape(outputShape, inferredShapes[0], m_ShapeInferenceMethod, "GatherLayer");
}

void GatherLayer::ExecuteStrategy(IStrategy& strategy) const
{
    strategy.ExecuteStrategy(this, GetParameters(), {}, GetName());
}

} // namespace armnn
