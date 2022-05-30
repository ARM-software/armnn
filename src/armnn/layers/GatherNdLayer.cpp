//
// Copyright © 2022 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "GatherNdLayer.hpp"
#include "LayerCloneBase.hpp"

#include <armnn/TypesUtils.hpp>
#include <armnn/backends/WorkloadData.hpp>
#include <armnn/backends/WorkloadFactory.hpp>

namespace armnn
{

GatherNdLayer::GatherNdLayer(const char* name)
    : Layer(2, 1, LayerType::GatherNd, name)
{
}

std::unique_ptr<IWorkload> GatherNdLayer::CreateWorkload(const armnn::IWorkloadFactory& factory) const
{
    GatherNdQueueDescriptor descriptor;
    SetAdditionalInfo(descriptor);

    return factory.CreateWorkload(LayerType::GatherNd, descriptor, PrepInfoAndDesc(descriptor));
}

GatherNdLayer* GatherNdLayer::Clone(Graph& graph) const
{
    return CloneBase<GatherNdLayer>(graph, GetName());
}

std::vector<TensorShape> GatherNdLayer::InferOutputShapes(const std::vector<TensorShape>& inputShapes) const
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

    // last dimension of indices
    unsigned int index_depth = indices[indicesDim - 1];
    ARMNN_ASSERT(index_depth <= paramsDim);

    // all but the last dimension of indices
    std::vector<unsigned int> outer_shape;
    outer_shape.reserve(indicesDim - 1);
    for (unsigned int i = 0; i < indicesDim - 1; ++i)
    {
        outer_shape.emplace_back(indices[i]);
    }

    // elements after index_depth
    std::vector<unsigned int> inner_shape;
    inner_shape.reserve(paramsDim - index_depth);
    for (unsigned int i = index_depth; i < paramsDim; ++i)
    {
        inner_shape.emplace_back(params[i]);
    }

    // concatenate outer_shape + inner_shape
    std::vector<unsigned int> output_shape;
    output_shape.reserve( outer_shape.size() + inner_shape.size() );
    output_shape.insert( output_shape.end(), outer_shape.begin(), outer_shape.end() );
    output_shape.insert( output_shape.end(), inner_shape.begin(), inner_shape.end() );

    const auto outputDim = static_cast<unsigned int>(output_shape.size());
    return std::vector<TensorShape>({ TensorShape({outputDim, output_shape.data()})});
}

void GatherNdLayer::ValidateTensorShapesFromInputs()
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

    ValidateAndCopyShape(outputShape, inferredShapes[0], m_ShapeInferenceMethod, "GatherNdLayer");
}

} // namespace armnn
