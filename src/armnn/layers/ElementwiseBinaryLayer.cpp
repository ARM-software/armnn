//
// Copyright © 2023 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "ElementwiseBinaryLayer.hpp"

#include "LayerCloneBase.hpp"

namespace armnn
{

ElementwiseBinaryLayer::ElementwiseBinaryLayer(const ElementwiseBinaryDescriptor& param, const char* name)
    : LayerWithParameters(2, 1, LayerType::ElementwiseBinary, param, name)
{
}

std::unique_ptr<IWorkload> ElementwiseBinaryLayer::CreateWorkload(const IWorkloadFactory& factory) const
{
    ElementwiseBinaryQueueDescriptor descriptor;
    SetAdditionalInfo(descriptor);

    return factory.CreateWorkload(LayerType::ElementwiseBinary, descriptor, PrepInfoAndDesc(descriptor));
}

ElementwiseBinaryLayer* ElementwiseBinaryLayer::Clone(Graph& graph) const
{
    return CloneBase<ElementwiseBinaryLayer>(graph, m_Param, GetName());
}

std::vector<TensorShape> ElementwiseBinaryLayer::InferOutputShapes(const std::vector<TensorShape>& inputShapes) const
{
    ARMNN_ASSERT(inputShapes.size() == 2);
    TensorShape input0 = inputShapes[0];
    TensorShape input1 = inputShapes[1];

    if (inputShapes[0].GetNumDimensions() < inputShapes[1].GetNumDimensions())
    {
        input1 = inputShapes[0];
        input0 = inputShapes[1];
    }

    unsigned int numDims     = input0.GetNumDimensions();
    unsigned int shiftedDims = input0.GetNumDimensions() - input1.GetNumDimensions();

    // Get the max of the inputs.
    std::vector<unsigned int> dims(numDims);
    for (unsigned int i = shiftedDims; i < numDims; i++)
    {
        unsigned int dim0 = input0[i];
        unsigned int dim1 = input1[i - shiftedDims];

        // Validate inputs are broadcast compatible.
        ARMNN_ASSERT_MSG(dim0 == dim1 || dim0 == 1 || dim1 == 1,
                         "Dimensions should either match or one should be of size 1.");

        dims[i] = std::max(dim0, dim1);
    }

    // Fill in the rest of the shifted dimensions.
    for (unsigned int i = 0; i < shiftedDims; i++)
    {
        dims[i] = input0[i];
    }

    return std::vector<TensorShape>({ TensorShape(numDims, dims.data()) });
}

void ElementwiseBinaryLayer::ValidateTensorShapesFromInputs()
{
    VerifyLayerConnections(2, CHECK_LOCATION());

    const TensorShape& outputShape = GetOutputSlot(0).GetTensorInfo().GetShape();

    VerifyShapeInferenceType(outputShape, m_ShapeInferenceMethod);

    auto inferredShapes = InferOutputShapes({ GetInputSlot(0).GetConnection()->GetTensorInfo().GetShape(),
                                              GetInputSlot(1).GetConnection()->GetTensorInfo().GetShape() });

    ARMNN_ASSERT(inferredShapes.size() == 1);

    ValidateAndCopyShape(outputShape, inferredShapes[0], m_ShapeInferenceMethod, GetLayerTypeAsCString(GetType()));
}

void ElementwiseBinaryLayer::ExecuteStrategy(IStrategy& strategy) const
{
    strategy.ExecuteStrategy(this, GetParameters(), {}, GetName());
}
} // namespace armnn
