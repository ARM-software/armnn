//
// Copyright © 2020 Arm Ltd and Contributors. All rights reserved.
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
    ARMNN_ASSERT(inputShapes.size() == 2);
    const TensorShape& input0 = inputShapes[0];
    const TensorShape& input1 = inputShapes[1];

    ARMNN_ASSERT(input0.GetNumDimensions() == input1.GetNumDimensions());
    unsigned int numDims = input0.GetNumDimensions();

    std::vector<unsigned int> dims(numDims);
    for (unsigned int i = 0; i < numDims; i++)
    {
        unsigned int dim0 = input0[i];
        unsigned int dim1 = input1[i];

        ARMNN_ASSERT_MSG(dim0 == dim1 || dim0 == 1 || dim1 == 1,
                         "Dimensions should either match or one should be of size 1.");

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
        GetInputSlot(0).GetConnection()->GetTensorInfo().GetShape(),
        GetInputSlot(1).GetConnection()->GetTensorInfo().GetShape()
    });
    ARMNN_ASSERT(inferredShapes.size() == 1);

    ValidateAndCopyShape(outputShape, inferredShapes[0], m_ShapeInferenceMethod, "LogicalBinaryLayer");
}

void LogicalBinaryLayer::ExecuteStrategy(IStrategy& strategy) const
{
    strategy.ExecuteStrategy(this, GetParameters(), {}, GetName());
}

} // namespace armnn
