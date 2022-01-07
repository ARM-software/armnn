//
// Copyright © 2017 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
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

std::vector<TensorShape> StridedSliceLayer::InferOutputShapes(
    const std::vector<TensorShape>& inputShapes) const
{
    ARMNN_ASSERT(inputShapes.size() == 1);

    TensorShape inputShape = inputShapes[0];
    std::vector<unsigned int> outputShape;
    unsigned int amountDimShrunk{0};

    for (unsigned int i = 0; i < inputShape.GetNumDimensions(); i++)
    {
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

        newSize = std::max(0, newSize);

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

    auto inferredShapes = InferOutputShapes({GetInputSlot(0).GetConnection()->GetTensorInfo().GetShape()});

    ARMNN_ASSERT(inferredShapes.size() == 1);

    ValidateAndCopyShape(outputShape, inferredShapes[0], m_ShapeInferenceMethod, "StridedSliceLayer");
}

ARMNN_NO_DEPRECATE_WARN_BEGIN
void StridedSliceLayer::Accept(ILayerVisitor& visitor) const
{
    visitor.VisitStridedSliceLayer(this, GetParameters(), GetName());
}
ARMNN_NO_DEPRECATE_WARN_END

} // namespace armnn
