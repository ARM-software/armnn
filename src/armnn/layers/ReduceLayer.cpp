//
// Copyright © 2020 Samsung Electronics Co Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "ReduceLayer.hpp"
#include "LayerCloneBase.hpp"

#include <armnn/TypesUtils.hpp>

#include <armnn/backends/WorkloadData.hpp>
#include <armnn/backends/WorkloadFactory.hpp>

namespace armnn
{

ReduceLayer::ReduceLayer(const ReduceDescriptor& param, const char* name)
    : LayerWithParameters(1, 1, LayerType::Reduce, param, name)
{
}

std::unique_ptr<IWorkload> ReduceLayer::CreateWorkload(const IWorkloadFactory& factory) const
{
    ReduceQueueDescriptor descriptor;
    descriptor.m_Parameters.m_vAxis           = m_Param.m_vAxis;
    descriptor.m_Parameters.m_KeepDims        = m_Param.m_KeepDims;
    descriptor.m_Parameters.m_ReduceOperation = m_Param.m_ReduceOperation;
    SetAdditionalInfo(descriptor);

    return factory.CreateWorkload(LayerType::Reduce, descriptor, PrepInfoAndDesc(descriptor));
}

ReduceLayer* ReduceLayer::Clone(Graph& graph) const
{
    auto layer =  CloneBase<ReduceLayer>(graph, m_Param, GetName());
    layer->m_Param.m_vAxis           = m_Param.m_vAxis;
    layer->m_Param.m_KeepDims        = m_Param.m_KeepDims;
    layer->m_Param.m_ReduceOperation = m_Param.m_ReduceOperation;

    return std::move(layer);
}

void ReduceLayer::ValidateTensorShapesFromInputs()
{
    VerifyLayerConnections(1, CHECK_LOCATION());

    const TensorShape& outputShape = GetOutputSlot(0).GetTensorInfo().GetShape();

    VerifyShapeInferenceType(outputShape, m_ShapeInferenceMethod);

    const TensorInfo& input = GetInputSlot(0).GetConnection()->GetTensorInfo();

    ARMNN_ASSERT_MSG(input.GetNumDimensions() > 0 && input.GetNumDimensions() <= 4,
                     "ReduceLayer: Reduce supports up to 4D input.");

    unsigned int rank = input.GetNumDimensions();
    unsigned int outputRank = 0;

    // Calculate output dimension
    if (m_Param.m_KeepDims)
    {
        outputRank = rank;
    }
    else if (m_Param.m_vAxis.empty())
    {
        outputRank = 1;
    }
    else if (m_Param.m_vAxis.size() > input.GetNumDimensions())
    {
        throw LayerValidationException("ReduceLayer: Dimensions to reduce can not be bigger than input dimensions");
    }
    else
    {
        outputRank = input.GetNumDimensions() - armnn::numeric_cast<unsigned int>(m_Param.m_vAxis.size());
        if (outputRank == 0)
        {
            outputRank = 1;
        }
    }

    std::vector<unsigned int> dimSizes(outputRank, 1);
    if (!m_Param.m_vAxis.empty())
    {
        // Skip the dimension that has been reduced unless keepDims is true.
        unsigned int outputIndex = 0;
        for (unsigned int i = 0; i < input.GetNumDimensions(); ++i)
        {
            if (std::find(m_Param.m_vAxis.begin(), m_Param.m_vAxis.end(), i) == m_Param.m_vAxis.end())
            {
                dimSizes[outputIndex] = armnn::numeric_cast<unsigned int>(input.GetShape()[i]);
                ++outputIndex;
            }
            else if (m_Param.m_KeepDims)
            {
                dimSizes[outputIndex] = 1;
                ++outputIndex;
            }
        }
    }
    const TensorShape& inferredShape = TensorShape(outputRank, dimSizes.data());

    ValidateAndCopyShape(outputShape, inferredShape, m_ShapeInferenceMethod, "ReduceLayer");
}

ARMNN_NO_DEPRECATE_WARN_BEGIN
void ReduceLayer::Accept(ILayerVisitor& visitor) const
{
    visitor.VisitReduceLayer(this, GetParameters(), GetName());
}
ARMNN_NO_DEPRECATE_WARN_END

} // namespace armnn
