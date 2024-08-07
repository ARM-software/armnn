//
// Copyright © 2020-2024 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "LogicalBinaryLayer.hpp"

#include "LayerCloneBase.hpp"

#include <armnn/backends/WorkloadData.hpp>
#include <armnn/backends/WorkloadFactory.hpp>

#include <algorithm>

namespace armnn
{

LogicalBinaryLayer::LogicalBinaryLayer(const LogicalBinaryDescriptor& param, const char* name)
    : LayerWithParameters(2, 1, LayerType::LogicalBinary, param, name)
{
}

std::unique_ptr<IWorkload> LogicalBinaryLayer::CreateWorkload(const IWorkloadFactory& factory) const
{
    LogicalBinaryQueueDescriptor descriptor;
    return factory.CreateWorkload(LayerType::LogicalBinary, descriptor, PrepInfoAndDesc(descriptor));
}

LogicalBinaryLayer* LogicalBinaryLayer::Clone(Graph& graph) const
{
    return CloneBase<LogicalBinaryLayer>(graph, m_Param, GetName());
}

std::vector<TensorShape> LogicalBinaryLayer::InferOutputShapes(const std::vector<TensorShape>& inputShapes) const
{
    if (inputShapes.size() != 2)
    {
        throw armnn::Exception("inputShapes' size is \"" + std::to_string(inputShapes.size()) +
                               "\" - should be \"2\".");
    }

    const TensorShape& input0 = inputShapes[0];
    const TensorShape& input1 = inputShapes[1];

    if (input0.GetNumDimensions() != input1.GetNumDimensions())
    {
        throw armnn::Exception("Input dimensions do not match (\""
                               + std::to_string(input0.GetNumDimensions()) +
                               "\" vs \""
                               + std::to_string(input1.GetNumDimensions()) + "\").");
    }

    unsigned int numDims = input0.GetNumDimensions();

    std::vector<unsigned int> dims(numDims);
    for (unsigned int i = 0; i < numDims; i++)
    {
        unsigned int dim0 = input0[i];
        unsigned int dim1 = input1[i];

        if (dim0 != dim1 && dim0 != 1 && dim1 != 1)
        {
            throw armnn::Exception("Dimensions should either match or one should be of size 1.");
        }

        dims[i] = std::max(dim0, dim1);
    }

    return std::vector<TensorShape>({ TensorShape(numDims, dims.data()) });
}

void LogicalBinaryLayer::ValidateTensorShapesFromInputs()
{
    VerifyLayerConnections(2, CHECK_LOCATION());

    const TensorShape& outputShape = GetOutputSlot(0).GetTensorInfo().GetShape();

    VerifyShapeInferenceType(outputShape, m_ShapeInferenceMethod);

    std::vector<TensorShape> inferredShapes = InferOutputShapes({
        GetInputSlot(0).GetTensorInfo().GetShape(),
        GetInputSlot(1).GetTensorInfo().GetShape()
    });

    if (inferredShapes.size() != 1)
    {
        throw armnn::LayerValidationException("inferredShapes has "
                                              + std::to_string(inferredShapes.size()) +
                                              " elements - should only have 1.");
    }

    ValidateAndCopyShape(outputShape, inferredShapes[0], m_ShapeInferenceMethod, "LogicalBinaryLayer");
}

void LogicalBinaryLayer::ExecuteStrategy(IStrategy& strategy) const
{
    strategy.ExecuteStrategy(this, GetParameters(), {}, GetName());
}

} // namespace armnn
