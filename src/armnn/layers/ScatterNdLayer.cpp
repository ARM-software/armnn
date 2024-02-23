//
// Copyright © 2024 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "ScatterNdLayer.hpp"
#include "LayerCloneBase.hpp"

#include <armnn/TypesUtils.hpp>
#include <armnn/backends/WorkloadData.hpp>
#include <armnn/backends/WorkloadFactory.hpp>

namespace armnn
{

ScatterNdLayer::ScatterNdLayer(const ScatterNdDescriptor &param, const char* name)
    : LayerWithParameters(3, 1, LayerType::ScatterNd, param, name)
{
}

std::unique_ptr<IWorkload> ScatterNdLayer::CreateWorkload(const armnn::IWorkloadFactory& factory) const
{
    ScatterNdQueueDescriptor descriptor;
    SetAdditionalInfo(descriptor);

    return factory.CreateWorkload(LayerType::ScatterNd, descriptor, PrepInfoAndDesc(descriptor));
}

ScatterNdLayer* ScatterNdLayer::Clone(Graph& graph) const
{
    auto layer = CloneBase<ScatterNdLayer>(graph, m_Param, GetName());

    return std::move(layer);
}

std::vector<TensorShape> ScatterNdLayer::InferOutputShapes(const std::vector<TensorShape>& inputShapes) const
{
    const auto inputDims = inputShapes[0].GetNumDimensions();

    std::vector<unsigned int> dimSizes(inputDims);

    for (unsigned i = 0;  i < inputDims; ++i)
    {
        dimSizes[i] = inputShapes[0][i];
    }

    TensorShape outputShape({ inputDims, dimSizes.data() });

    return std::vector<TensorShape>({ outputShape });
}

void ScatterNdLayer::ValidateTensorShapesFromInputs()
{
    VerifyLayerConnections(3, CHECK_LOCATION());

    const TensorShape& outputShape = GetOutputSlot(0).GetTensorInfo().GetShape();

    VerifyShapeInferenceType(outputShape, m_ShapeInferenceMethod);

    if (m_Param.m_InputEnabled)
    {
        std::vector<TensorShape> inferredShapes = InferOutputShapes(
            {GetInputSlot(0).GetTensorInfo().GetShape(),
             GetInputSlot(1).GetTensorInfo().GetShape(),
             GetInputSlot(2).GetTensorInfo().GetShape()});

        if (inferredShapes.size() != 1) {
            throw armnn::LayerValidationException("inferredShape has " +
                                                  std::to_string(inferredShapes.size()) +
                                                  " elements - should only have 1.");
        }

        ValidateAndCopyShape(outputShape, inferredShapes[0], m_ShapeInferenceMethod, "ScatterLayer");
    }
    else
    {
        // No input tensor, only shape provided via input slot
        // In this case, we cannot validate the output shape from the input shape, but we can
        // validate that the dimensions of shape and output tensor matched
        unsigned int shapeDims = GetInputSlot(0).GetTensorInfo().GetNumDimensions();
        unsigned int outputDims = GetOutputSlot(0).GetTensorInfo().GetNumDimensions();

        if (shapeDims != outputDims)
        {
            throw armnn::LayerValidationException("shape dimension " +
                                                  std::to_string(shapeDims) +
                                                  " and output dimension " +
                                                  std::to_string(outputDims) +
                                                  " are not matched.");
        }
    }
}

} // namespace armnn
