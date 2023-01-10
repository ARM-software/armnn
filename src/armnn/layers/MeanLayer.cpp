//
// Copyright © 2017-2023 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "MeanLayer.hpp"
#include "LayerCloneBase.hpp"

#include <armnn/utility/NumericCast.hpp>

#include <armnn/backends/TensorHandle.hpp>
#include <armnn/backends/WorkloadData.hpp>
#include <armnn/backends/WorkloadFactory.hpp>

#include <cstring>

namespace armnn
{

MeanLayer::MeanLayer(const armnn::MeanDescriptor& param, const char* name)
    : LayerWithParameters(1, 1, LayerType::Mean, param, name)
{}

std::unique_ptr<IWorkload> MeanLayer::CreateWorkload(const armnn::IWorkloadFactory& factory) const
{
    MeanQueueDescriptor descriptor;
    descriptor.m_Parameters.m_Axis = m_Param.m_Axis;
    descriptor.m_Parameters.m_KeepDims = m_Param.m_KeepDims;
    SetAdditionalInfo(descriptor);

    return factory.CreateWorkload(LayerType::Mean, descriptor, PrepInfoAndDesc(descriptor));
}

MeanLayer* MeanLayer::Clone(Graph& graph) const
{
    auto layer = CloneBase<MeanLayer>(graph, m_Param, GetName());

    layer->m_Param.m_Axis = m_Param.m_Axis;
    layer->m_Param.m_KeepDims = m_Param.m_KeepDims;

    return std::move(layer);
}

void MeanLayer::ValidateTensorShapesFromInputs()
{
    VerifyLayerConnections(1, CHECK_LOCATION());

    const TensorShape& outputShape = GetOutputSlot(0).GetTensorInfo().GetShape();

    VerifyShapeInferenceType(outputShape, m_ShapeInferenceMethod);

    std::vector<TensorShape> inferredShapes = InferOutputShapes(
            { GetInputSlot(0).GetConnection()->GetTensorInfo().GetShape() });

    ARMNN_ASSERT(inferredShapes.size() == 1);
    ARMNN_ASSERT(inferredShapes[0].GetDimensionality() == Dimensionality::Specified);

    ValidateAndCopyShape(outputShape, inferredShapes[0], m_ShapeInferenceMethod, "MeanLayer");
}

std::vector<TensorShape> MeanLayer::InferOutputShapes(const std::vector<TensorShape>& inputShapes) const
{
    ARMNN_ASSERT(inputShapes.size() == 1);
    const TensorShape& input = inputShapes[0];

    ARMNN_ASSERT_MSG(input.GetNumDimensions() > 0 && input.GetNumDimensions() <= 4,
                     "MeanLayer: Mean supports up to 4D input.");

    unsigned int rank = input.GetNumDimensions();
    unsigned int outputRank = 0;

    // Calculate output dimension
    if (m_Param.m_KeepDims)
    {
        outputRank = rank;
    }
    else if (m_Param.m_Axis.empty())
    {
        outputRank = 1;
    }
    else if (m_Param.m_Axis.size() > input.GetNumDimensions())
    {
        throw LayerValidationException("MeanLayer: Dimensions to reduce can not be bigger than input dimensions");
    }
    else
    {
        outputRank = input.GetNumDimensions() - armnn::numeric_cast<unsigned int>(m_Param.m_Axis.size());
        if (outputRank == 0)
        {
            outputRank = 1;
        }
    }

    std::vector<unsigned int> dimSizes(outputRank, 1);
    if (!m_Param.m_Axis.empty())
    {
        // Skip the dimension that has been reduced unless keepDims is true.
        unsigned int outputIndex = 0;
        for (unsigned int i = 0; i < input.GetNumDimensions(); ++i)
        {
            if (std::find(m_Param.m_Axis.begin(), m_Param.m_Axis.end(), i) == m_Param.m_Axis.end())
            {
                dimSizes[outputIndex] = armnn::numeric_cast<unsigned int>(input[i]);
                ++outputIndex;
            }
            else if (m_Param.m_KeepDims)
            {
                dimSizes[outputIndex] = 1;
                ++outputIndex;
            }
        }
    }
    return std::vector<TensorShape>({ TensorShape(outputRank, dimSizes.data()) });
}

void MeanLayer::ExecuteStrategy(IStrategy& strategy) const
{
    strategy.ExecuteStrategy(this, GetParameters(), {}, GetName());
}

} // namespace armnn
