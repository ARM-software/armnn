//
// Copyright © 2018-2024 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//
// Copyright © 2018 The TensorFlow Authors. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0
//
#include "StridedSliceLayer.hpp"

#include "LayerCloneBase.hpp"

#include <armnn/utility/NumericCast.hpp>

#include <armnn/backends/WorkloadData.hpp>
#include <armnn/backends/WorkloadFactory.hpp>

namespace armnn
{

StridedSliceLayer::StridedSliceLayer(const armnn::StridedSliceDescriptor& param, const char* name)
    : LayerWithParameters(1, 1, LayerType::StridedSlice, param, name)
{
}

std::unique_ptr<IWorkload> StridedSliceLayer::CreateWorkload(const IWorkloadFactory& factory) const
{
    StridedSliceQueueDescriptor descriptor;

    descriptor.m_Parameters.m_Begin  = m_Param.m_Begin;
    descriptor.m_Parameters.m_End    = m_Param.m_End;
    descriptor.m_Parameters.m_Stride = m_Param.m_Stride;

    // Optional parameters
    descriptor.m_Parameters.m_BeginMask      = m_Param.m_BeginMask;
    descriptor.m_Parameters.m_EndMask        = m_Param.m_EndMask;
    descriptor.m_Parameters.m_EllipsisMask   = m_Param.m_EllipsisMask;
    descriptor.m_Parameters.m_NewAxisMask    = m_Param.m_NewAxisMask;
    descriptor.m_Parameters.m_ShrinkAxisMask = m_Param.m_ShrinkAxisMask;

    SetAdditionalInfo(descriptor);

    return factory.CreateWorkload(LayerType::StridedSlice, descriptor, PrepInfoAndDesc(descriptor));
}

StridedSliceLayer* StridedSliceLayer::Clone(Graph& graph) const
{
    return CloneBase<StridedSliceLayer>(graph, m_Param, GetName());
}

// Content in this function (fixes related to NewAxisMask and EllipsisMask) are paraphrased from:
// tensorflow/tensorflow/lite/kernels/strided_slice.cc from the function BuildStridedSliceParams
std::vector<TensorShape> StridedSliceLayer::InferOutputShapes(
    const std::vector<TensorShape>& inputShapes) const
{
    if (inputShapes.size() != 1)
    {
        throw armnn::Exception("inputShapes' size is \"" + std::to_string(inputShapes.size()) +
                               "\" - should be \"1\".");
    }

    TensorShape inputShape = inputShapes[0];
    std::vector<unsigned int> outputShape;
    unsigned int amountDimShrunk{0};

    // Getting the actual number of output dimensions, including axes added with the NewAxisMask
    unsigned int outputDims = inputShape.GetNumDimensions();
    for(unsigned int i = 0; i < m_Param.m_Begin.size(); ++i)
    {
        // Adding to dimension count for every set bit of NewAxisMask not covered by the EllipsisMask
        if(m_Param.m_NewAxisMask & (1 << i) && !(m_Param.m_EllipsisMask & (1 << i)))
        {
            ++outputDims;
        }
    }

    // Modifying the EllipsisMask based on the NewAxisMask (expand for any newly added axes)
    // and the NewAxisMask based on the EllipsisMask (offset based on the expanded ellipsis)
    int realEllipsisMask = 0, realNewAxisMask = 0;
    // The number of bits the ellipsis mask was expanded by
    unsigned int ellipsisExpandedBy = 0;
    for(unsigned int i = 0; i < outputDims; ++i)
    {
        if(m_Param.m_EllipsisMask & (1 << i))
        {
            // The end index of the expanded ellipsis mask (start is at i)
            // End Index calculation - i+1 (for non-expanded ellipsis) + outputDims-inputDims (number of added dims)
            unsigned int endIdx = std::min(i + 1u + outputDims - inputShape.GetNumDimensions(), outputDims);

            // Calculation: the total size of the mask -1 for the already existing bit in the original mask
            ellipsisExpandedBy = endIdx - i - 1;

            // Setting mask bit to 1 for the entire expanded ellipsis
            for(; i < endIdx; ++i)
            {
                realEllipsisMask |= (1 << i);
            }
        }

        // Setting the real NewAxisMask based on the expanded ellipsis size
        if(m_Param.m_NewAxisMask & (1 << (i - ellipsisExpandedBy)))
        {
            realNewAxisMask |= (1 << i);
        }
    }

    // The backwards offset by which i is ahead of the actual inputTensor dimension
    unsigned int inputDimOffset = 0;
    // Iterating through the parameters and inferring output shape
    for (unsigned int i = 0; i < outputDims; ++i)
    {
        // Add entire dimension if EllipsisMask is set
        if(realEllipsisMask & (1 << i))
        {
            outputShape.push_back(inputShape[i - inputDimOffset]);
            continue;
        }
        // Add dimension of length 1 if NewAxisMask is set
        if(realNewAxisMask & (1 << i))
        {
            outputShape.push_back(1);
            ++inputDimOffset;
            continue;
        }
        // Fill the rest of the inferred shape (dimensions greater than the input shape)
        if(i >= inputShape.GetNumDimensions())
        {
            // If EllipsisMask was set at any point, the TensorFlow behavior is to fill the rest of the tensor with 1
            // Otherwise, the remaining dimensions from the inputShape (which were skipped over) are used
            if(realEllipsisMask > 0)
            {
                outputShape.push_back(1);
            }
            else
            {
                outputShape.push_back(inputShape[i - inputDimOffset]);
            }
            continue;
        }

        int stride = m_Param.m_Stride[i];
        int start = m_Param.GetStartForAxis(inputShape, i);
        int stop = m_Param.GetStopForAxis(inputShape, i, start);

        if (m_Param.m_ShrinkAxisMask & (1 << i))
        {
            amountDimShrunk+=1;

            // If the difference between the start point and the end point of the slice on an axis being shrunk
            // is greater than 1 then throw an error as the output will not be large enough to hold the slice
            if (((m_Param.m_Begin[i] - m_Param.m_End[i]) > 1) || ((m_Param.m_Begin[i] - m_Param.m_End[i]) < -1))
            {
                throw LayerValidationException(
                    "StridedSlice: Attempting to take a larger slice than can fit in inferred output");
            }

            if (stride < 0)
            {
                throw LayerValidationException(
                    "StridedSlice: Stride can not be negative with Shrink Axis Mask set.");
            }
            continue;
        }

        int newSize = stride > 0 ? ((stop - start) + stride - 1) / stride :
                                   ((start - stop) - stride - 1) / -stride;

        // Making sure the dimension size doesn't go out of bounds
        newSize = std::max(0, newSize);
        newSize = std::min(newSize, armnn::numeric_cast<int>(inputShape[i - inputDimOffset]));

        outputShape.push_back(armnn::numeric_cast<unsigned int>(newSize));
    }

    if (outputShape.size() == 0 && (inputShape.GetNumDimensions() - amountDimShrunk) == 0)
    {
        outputShape.push_back(1);
    }

    return std::vector<TensorShape>({
        TensorShape(armnn::numeric_cast<unsigned int>(outputShape.size()), &outputShape[0]) });
}

void StridedSliceLayer::ValidateTensorShapesFromInputs()
{
    VerifyLayerConnections(1, CHECK_LOCATION());

    const TensorShape& outputShape = GetOutputSlot(0).GetTensorInfo().GetShape();

    VerifyShapeInferenceType(outputShape, m_ShapeInferenceMethod);

    auto inferredShapes = InferOutputShapes({GetInputSlot(0).GetTensorInfo().GetShape()});

    if (inferredShapes.size() != 1)
    {
        throw armnn::LayerValidationException("inferredShapes has "
                                              + std::to_string(inferredShapes.size()) +
                                              " elements - should only have 1.");
    }

    ValidateAndCopyShape(outputShape, inferredShapes[0], m_ShapeInferenceMethod, "StridedSliceLayer");
}

void StridedSliceLayer::ExecuteStrategy(IStrategy& strategy) const
{
    strategy.ExecuteStrategy(this, GetParameters(), {}, GetName());
}

} // namespace armnn
