//
// Copyright © 2017 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "ElementwiseBaseLayer.hpp"

#include "InternalTypes.hpp"
#include "armnn/Exceptions.hpp"
#include <armnn/TypesUtils.hpp>
#include <armnn/utility/Assert.hpp>

namespace armnn
{

ElementwiseBaseLayer::ElementwiseBaseLayer(unsigned int numInputSlots,
                                           unsigned int numOutputSlots,
                                           LayerType type,
                                           const char* name)
    : Layer(numInputSlots, numOutputSlots, type, name)
{}

std::vector<TensorShape> ElementwiseBaseLayer::InferOutputShapes(const std::vector<TensorShape>& inputShapes) const
{
    ARMNN_ASSERT(inputShapes.size() == 2);
    TensorShape input0 = inputShapes[0];
    TensorShape input1 = inputShapes[1];

    if (m_ShapeInferenceMethod == ShapeInferenceMethod::ValidateOnly)
    {
        if (input0.GetNumDimensions() != input1.GetNumDimensions())
        {
            std::stringstream errorMessage;
            errorMessage << GetLayerTypeAsCString(GetType()) << " layer \"" << GetName() << "\": ";
            errorMessage << "The tensor inputs to an element-wise operator are expected to have equal number of "
                            "dimensions. First = "
                         << input0.GetNumDimensions() << " second = " << input1.GetNumDimensions();
            throw InvalidArgumentException(errorMessage.str(), CHECK_LOCATION());
        }
    }
    else if (m_ShapeInferenceMethod == ShapeInferenceMethod::InferAndValidate &&
             inputShapes[0].GetNumDimensions() < inputShapes[1].GetNumDimensions())
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

#if !NDEBUG
        // Validate inputs are broadcast compatible.
        ARMNN_ASSERT_MSG(dim0 == dim1 || dim0 == 1 || dim1 == 1,
                         "Dimensions should either match or one should be of size 1.");
#endif

        dims[i] = std::max(dim0, dim1);
    }

    // Fill in the rest of the shifted dimensions.
    for (unsigned int i = 0; i < shiftedDims; i++)
    {
        dims[i] = input0[i];
    }

    return std::vector<TensorShape>({ TensorShape(numDims, dims.data()) });
}

void ElementwiseBaseLayer::ValidateTensorShapesFromInputs()
{
    VerifyLayerConnections(2, CHECK_LOCATION());

    const TensorShape& outputShape = GetOutputSlot(0).GetTensorInfo().GetShape();

    VerifyShapeInferenceType(outputShape, m_ShapeInferenceMethod);

    auto inferredShapes = InferOutputShapes({ GetInputSlot(0).GetConnection()->GetTensorInfo().GetShape(),
                                              GetInputSlot(1).GetConnection()->GetTensorInfo().GetShape() });

    ARMNN_ASSERT(inferredShapes.size() == 1);

    ValidateAndCopyShape(outputShape, inferredShapes[0], m_ShapeInferenceMethod, GetLayerTypeAsCString(GetType()));
}

void ElementwiseBaseLayer::ExecuteStrategy(IStrategy& strategy) const
{
    strategy.ExecuteStrategy(this, BaseDescriptor(), {}, GetName());
}

}    // namespace armnn
