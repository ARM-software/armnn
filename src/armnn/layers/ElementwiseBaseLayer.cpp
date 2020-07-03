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

ElementwiseBaseLayer::ElementwiseBaseLayer(unsigned int numInputSlots, unsigned int numOutputSlots,
                                           LayerType type, const char* name)
    : Layer(numInputSlots, numOutputSlots, type, name)
{
}

std::vector<TensorShape> ElementwiseBaseLayer::InferOutputShapes(const std::vector<TensorShape>& inputShapes) const
{
    ARMNN_ASSERT(inputShapes.size() == 2);
    auto& input0 = inputShapes[0];
    auto& input1 = inputShapes[1];

    // Get the max of the inputs.
    ARMNN_ASSERT(input0.GetNumDimensions() == input1.GetNumDimensions());
    unsigned int numDims = input0.GetNumDimensions();
    std::vector<unsigned int> dims(numDims);

    for (unsigned int i = 0; i < numDims; i++)
    {
        unsigned int dim0 = input0[i];
        unsigned int dim1 = input1[i];

#if !NDEBUG
        // Validate inputs are broadcast compatible.
        ARMNN_ASSERT_MSG(dim0 == dim1 || dim0 == 1 || dim1 == 1,
                         "Dimensions should either match or one should be of size 1.");
#endif

        dims[i] = std::max(dim0, dim1);
    }

    return std::vector<TensorShape>({ TensorShape(numDims, dims.data()) });
}

void ElementwiseBaseLayer::ValidateTensorShapesFromInputs(ShapeInferenceMethod shapeInferenceMethod)
{
    VerifyLayerConnections(2, CHECK_LOCATION());

    const TensorShape& outputShape = GetOutputSlot(0).GetTensorInfo().GetShape();

    VerifyShapeInferenceType(outputShape, shapeInferenceMethod);

    auto inferredShapes = InferOutputShapes({
        GetInputSlot(0).GetConnection()->GetTensorInfo().GetShape(),
        GetInputSlot(1).GetConnection()->GetTensorInfo().GetShape()
    });

    ARMNN_ASSERT(inferredShapes.size() == 1);

    ValidateAndCopyShape(outputShape, inferredShapes[0], shapeInferenceMethod, GetLayerTypeAsCString(GetType()));
}

} // namespace armnn
