//
// Copyright © 2023 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "TileLayer.hpp"

#include <armnn/backends/WorkloadFactory.hpp>
#include "layers/LayerCloneBase.hpp"

namespace armnn
{
TileLayer::TileLayer(const TileDescriptor &param, const char *name)
    : LayerWithParameters(1, 1, LayerType::Tile, param, name)
{}

std::unique_ptr<IWorkload> TileLayer::CreateWorkload(const IWorkloadFactory &factory) const
{
    TileQueueDescriptor descriptor;
    SetAdditionalInfo(descriptor);

    return factory.CreateWorkload(LayerType::Tile, descriptor, PrepInfoAndDesc(descriptor));
}

TileLayer* TileLayer::Clone(armnn::Graph &graph) const
{
    auto layer = CloneBase<TileLayer>(graph, m_Param, GetName());

    return std::move(layer);
}

std::vector<TensorShape> TileLayer::InferOutputShapes(const std::vector<TensorShape>& inputShapes) const
{
    ARMNN_ASSERT(inputShapes.size() == 1);
    const TensorShape& inputShape = inputShapes[0];

    uint32_t numberOfDimensions = inputShape.GetNumDimensions();
    std::vector<unsigned int> dimensionSizes;
    dimensionSizes.reserve(numberOfDimensions);

    // Check input shape and multiples have same length and multiply them together to get output shape
    if(numberOfDimensions == m_Param.m_Multiples.size())
    {
        for(uint32_t i = 0; i < numberOfDimensions; ++i)
        {
            dimensionSizes.emplace_back(inputShape[i] * m_Param.m_Multiples[i]);
        }
    }
    else
    {
        throw LayerValidationException("TileLayer: input rank and multiples length are different.");
    }

    return std::vector<TensorShape>({TensorShape({numberOfDimensions, dimensionSizes.data()})});
}

void TileLayer::ValidateTensorShapesFromInputs()
{
    VerifyLayerConnections(1, CHECK_LOCATION());

    const TensorShape& outputShape = GetOutputSlot(0).GetTensorInfo().GetShape();

    VerifyShapeInferenceType(outputShape, m_ShapeInferenceMethod);

    auto inferredShapes = InferOutputShapes({ GetInputSlot(0).GetTensorInfo().GetShape() });

    ARMNN_ASSERT(inferredShapes.size() == 1);

    ValidateAndCopyShape(outputShape, inferredShapes[0], m_ShapeInferenceMethod, "TileLayer");
}

}